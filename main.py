import os, io, zipfile
import shutil
import asyncio
import time
import cv2
from fastapi import Body, Depends, FastAPI, Security, UploadFile, File, HTTPException, Request, APIRouter, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uuid
import base64
from datetime import datetime, timedelta
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from core.ec_controller.ec_verification_controller import EcVerification
from core.ec_controller.model.ec_verification_model import ApiAuthRequest, EcVerificationRequest, Identify, Verify
from utils.hitCounter_helper import HitCounter
from core.ocr_controller.face_matcher_controller import ArcFaceMatcher
from core.ocr_controller.ocr_revanced_controller import OCRProcessor

from typing import Optional

from utils.auth_helper import API_KEY, create_token, verify_jwt_token
from utils.response_model import APIResponse

_show_steps = True

app = FastAPI(
        title="Project OCR",
        description="API documentation",
        version="3.0.0",
        contact={
            "name": "Agent Banking Team",
            "email": "agent_banking_team@datasoft-bd.com",
        }
    )

protected_router = APIRouter(dependencies=[Depends(verify_jwt_token)])
hit_counter = HitCounter()


# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create limiter instance
limiter = Limiter(key_func=get_remote_address)

# Apply limiter to FastAPI app
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return APIResponse.error(
        message="Rate limit exceeded. Please try again later.",
        code=429,
        details=str(exc.detail)  # optional
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

# Initialize the OCR processor
processor = OCRProcessor()
faceMatcher = ArcFaceMatcher()

# Semaphore to limit concurrent tasks
semaphore = asyncio.Semaphore(2)

async def process_image(image_side: str, file: UploadFile, unique_id:str, profile: UploadFile, face_match_enable: bool=False) -> dict:
    try:
        # Save the uploaded file temporarily
        temp_dir = f"./history/{unique_id}"
        temp_dir_by_side = f"{temp_dir}/{image_side}"
        os.makedirs(temp_dir_by_side, exist_ok=True)
        temp_file_path = os.path.join(temp_dir_by_side, f'{file.filename}')
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        
        if profile:
            temp_profile_path = os.path.join(temp_dir_by_side, f'Profile-{profile.filename}')
            with open(temp_profile_path, "wb") as temp_file:
                temp_file.write(await profile.read())
        else:
            temp_profile_path = None

        # Process the image with a semaphore
        async with semaphore:
            result = await processor.letsStart(temp_file_path, unique_id, image_side, temp_profile_path, _show_steps)
        
        if not _show_steps:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing {image_side}: {str(e)}")

async def create_folder_if_not_exists(folder_name):

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created in {current_directory}.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting...")
    asyncio.create_task(create_folder_if_not_exists('history'))
    yield
    print("Application is shutting down...")

@app.get("/", include_in_schema=False)
async def initial():
    return {
        "message": "Ocr Service is Running....",
        "swagger": "/docs"
    }

@app.post("/api-auth", tags=["Auth API's"])
@limiter.limit("10/minute")
async def api_auth(
    request: Request,
    request_body: ApiAuthRequest = Body(...),
):
    if request_body.apiKey != API_KEY:
        return APIResponse.error(
            message="Invalid API key",
            code=403,
            details="Provided API key does not match the expected value."
        )

    payload = {
        "exp": datetime.utcnow() + timedelta(hours=1),
        "ec_credentials_valid":False
    }

    if request_body.baseUrl and request_body.username and request_body.password:
        ec = EcVerification(
            baseUrl=request_body.baseUrl,
            username=request_body.username,
            password=request_body.password
        )

        try:
            token_response = await ec.get_access_token()
            if token_response["status"] == 200:
                payload.update({
                    "ec_credentials_valid": True,
                    "ec_baseurl": request_body.baseUrl,
                    "ec_username": request_body.username,
                    "ec_password": request_body.password,
                    "ec_token": token_response["access_token"]
                })
        except Exception as e:
            payload["ec_credentials_valid"] = False

    jwt_token = create_token(payload)

    return APIResponse.success(
        message="Token created successfully",
        data={
            "access_token": jwt_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "ec_authenticated": payload["ec_credentials_valid"]
        }
    )

@protected_router.post("/image-to-text", tags=["OCR API's"])
@limiter.limit("10/minute")
async def image_to_text_extract(
    request: Request,
    image: UploadFile = File(..., description="Upload an image file")
):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="Image is required.")
        
        contents = await image.read()
        
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as f:
            f.write(contents)
        
        result = await processor.direct_extract_text_from_image(temp_image_path)

        # Clean up the temporary file
        import os
        os.remove(temp_image_path)

        return APIResponse.success(
                data=result
            )

    except Exception as e:
        return APIResponse.error(
                    message="Unable to Extract Text From Image",
                    code=400,
                    details=f"Error processing image: {e}"
                )


@protected_router.post("/face-matching", tags=["Image Quality API's"])
@limiter.limit("1/5 seconds")
@limiter.limit("10/minute")
async def face_matching(
    request: Request,
    image1: UploadFile = File(..., description="Front side image"),
    image2: UploadFile = File(..., description="Profile image")
):
    unique_id = str(uuid.uuid4())
    # Save the uploaded file temporarily
    temp_dir = f"./history/fm_{unique_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    image1_path = os.path.join(temp_dir, f'{image1.filename}')
    with open(image1_path, "wb") as temp_file:
        temp_file.write(await image1.read())

    
    image2_path = os.path.join(temp_dir, f'{image2.filename}')
    with open(image2_path, "wb") as temp_file:
        temp_file.write(await image2.read())
        
    faceMatchingResult = await faceMatcher.compare(image1_path, image2_path)

    if _show_steps:
        image = await asyncio.to_thread(cv2.imread, image1_path)
        if faceMatchingResult.get("profileFace") is not None and faceMatchingResult.get("profileFace").any():
            combined_img = await processor.process_images_into_one(image, [faceMatchingResult.get("profileFace"),faceMatchingResult.get("nidFace")])  
        else:
            image2 = await asyncio.to_thread(cv2.imread, image2_path)
            combined_img = await processor.process_images_into_one(image, [image2])    
            
        combined_img = await processor.write_text_to_image(
            combined_img, 10, 0, 0, 0, f"Result (Matched-{faceMatchingResult['is_face_matched']})"
        )
        combined_img = await processor.write_text_to_image(
            combined_img, 10, 10, 0, 0, f"(Score-{faceMatchingResult['face_matched_score']}) (TScore-{faceMatchingResult['face_matched_threshold']})"
        )
        output_path = os.path.join(f"./history/{unique_id}", f"Faces.jpg")
        await asyncio.to_thread(cv2.imwrite, output_path, combined_img)
        
    return APIResponse.success(data={
            "is_face_matched": faceMatchingResult["is_face_matched"],
            "face_matched_threshold": faceMatchingResult["face_matched_threshold"],
            "face_matched_score": faceMatchingResult["face_matched_score"]
        })


@protected_router.post("/profile-image-quality", tags=["Image Quality API's"])
@limiter.limit("10/minute")
async def check_profile_picture_quality(
    request: Request,
    image: UploadFile = File(...)
):
    # Save temporarily to disk (or use in-memory if preferred)
    image_bytes = await image.read()
    result = await asyncio.to_thread(faceMatcher.validate_face_image_sync, image_bytes)
    return APIResponse.success(
            data=result
        )


@protected_router.post("/image-quality", tags=["Image Quality API's"])
@limiter.limit("15/minute")
async def evaluate_nidfront_image_quality(
    request: Request,
    image: UploadFile = File(..., description="Upload an image file"),
    high_accuracy: bool = Form(False, description="Enable high accuracy mode"),
    is_nid_dob_extractable: bool = Form(False, description="With fields checking"),
):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="Image is required.")

        unique_id = str(uuid.uuid4())
        temp_dir = f"./history/{unique_id}"
        os.makedirs(temp_dir, exist_ok=True)
        contents = await image.read()
        temp_image_path = os.path.join(temp_dir, f'image-{image.filename}')
        with open(temp_image_path, "wb") as f:
            f.write(contents)

        quality_response = await processor.estimate_image_quality(temp_image_path, unique_id, high_accuracy, is_nid_dob_extractable)

        shutil.rmtree(temp_dir)

        return APIResponse.success(
                data=quality_response
            )

    except Exception as e:
        return APIResponse.error(
                    message="Unable to Extract Text From Image",
                    code=400,
                    details=f"Error processing image: {e}"
                )

async def handle_ocr_request(request: Request, profile: UploadFile, front_side: UploadFile) -> dict:
    if not front_side:
        raise HTTPException(status_code=400, detail="Front side image is required.")

    unique_id = str(uuid.uuid4())

    x_forwarded_for = request.headers.get('X-Forwarded-For')
    client_ip = x_forwarded_for.split(',')[0] if x_forwarded_for else request.client.host
    print(f"Executing New Request, ID: {unique_id}, From IP: {client_ip}")
    
    # Create OCR task(s)
    tasks = [
        process_image("FrontSide", front_side, unique_id, profile)
    ]

    # Run tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    response = {}
    for result in results:
        if isinstance(result, dict):
            response.update(result)
        elif isinstance(result, Exception):
            response["error"] = str(result)

    response["requestId"] = unique_id
    await hit_counter.log_hit(result)
    return response

@protected_router.post("/process-ocr", tags=["OCR API's"])
@limiter.limit("1/5 seconds")
@limiter.limit("10/minute")
async def process_ocr_only(
    request: Request,
    front_side: UploadFile = File(..., description="Front side image"),
    profile: Optional[UploadFile] = File(None, description="Profile image")
):
    result = await handle_ocr_request(request, profile, front_side)
    if result.get("nid_data"):
        return APIResponse.success(data=result)
    # Enhanced error detection and logging
    error_message = result.get("message", "OCR failed. Unknown error.")
    error_details = result.get("error", None)
    # Log the error for debugging
    print(f"[OCR ERROR] /process-ocr failed: message={error_message}, details={error_details}")
    # Try to provide more specific error messages
    if error_details:
        if isinstance(error_details, dict):
            details_str = error_details.get("details", str(error_details))
        else:
            details_str = str(error_details)
        if "tesseract" in details_str.lower() and "ben.traineddata" in details_str.lower():
            error_message = "Bengali language data (ben.traineddata) for Tesseract is missing. Please install it as per the README."
        elif "tesseract" in details_str.lower():
            error_message = "Tesseract OCR engine is not installed or not found. Please install Tesseract and ensure it is in your PATH."
        elif "forbidden" in details_str.lower() or "403" in details_str:
            error_message = "You are not authorized to access this resource. Please check your credentials or permissions."
    return APIResponse.error(error_message, 406, error_details)


@protected_router.post("/process-ocr-with-ec", tags=["OCR API's"])
@limiter.limit("1/5 seconds")
@limiter.limit("10/minute")
async def process_ocr_with_ec(
    request: Request,
    token_data: dict = Security(verify_jwt_token),
    front_side: UploadFile = File(..., description="Front side image"),
    profile: UploadFile = File(..., description="Profile image")
):
    
    start_time = time.time() 
    result = await handle_ocr_request(request, profile, front_side)
    if(token_data.get("ec_credentials_valid")):
        ec = EcVerification(
            baseUrl=token_data.get("ec_baseurl"),
            username=token_data.get("ec_username"), # "MMBL.ABD",
            password=token_data.get("ec_password"),   # "Mmbplc@1234"
        )
        nid_data = result["nid_data"]
        request_body = EcVerificationRequest(
            identify=Identify(
                nationalId=nid_data.get("nationalId")
            ),
            verify=Verify(
                name=nid_data.get("name"),
                nameEn=nid_data.get("nameEn"),
                father=nid_data.get("father"),
                mother=nid_data.get("mother"),
                dateOfBirth=nid_data.get("dateOfBirth")
            )
        )
        verification_result = await ec.verify_voter(request_body)
        if verification_result["status"] == 200:
            result["ec_data"] = verification_result["result"]["ec_data"]
            unique_id = result["requestId"]
            # Save the uploaded file temporarily
            temp_dir = f"./history/{unique_id}/FrontSide"

            if verification_result["result"]["ec_data"]["photo"]:
                image1 = base64.b64decode(verification_result["result"]["ec_data"]["photo"])
                image1_path = os.path.join(temp_dir, f'ec_face.png')
                with open(image1_path, "wb") as temp_file:
                    temp_file.write(image1)
                    
                image2_path = os.path.join(temp_dir, f'{front_side.filename}')

                faceMatchingResult = await faceMatcher.compare(image1_path, image2_path)

                if _show_steps:
                    image = await asyncio.to_thread(cv2.imread, image1_path)
                    if faceMatchingResult.get("profileFace") is not None and faceMatchingResult.get("profileFace").any():
                        combined_img = await processor.process_images_into_one(image, [faceMatchingResult.get("profileFace"),faceMatchingResult.get("nidFace")])  
                    
                    combined_img = await processor.write_text_to_image(
                        combined_img, 10, 0, 0, 0, f"Result (Matched-{faceMatchingResult['is_face_matched']})"
                    )
                    combined_img = await processor.write_text_to_image(
                        combined_img, 10, 10, 0, 0, f"(Score-{faceMatchingResult['face_matched_score']}) (TScore-{faceMatchingResult['face_matched_threshold']})"
                    )
                    output_path = os.path.join(temp_dir, f"Faces.jpg")
                    await asyncio.to_thread(cv2.imwrite, output_path, combined_img)
                    result["face_data"]={
                        "is_face_matched":faceMatchingResult["is_face_matched"],
                        "face_matched_score":faceMatchingResult["face_matched_score"],
                        "face_matched_threshold":faceMatchingResult["face_matched_threshold"]
                    }
        else:
            result["ec_data"] = {}
    else:
       result["ec_data"] = {}

    
    end_time = time.time() 
    total_time = round(end_time - start_time, 2)
    result['total_time_in_seconds'] = total_time
    # You can add EC verification logic here if needed in future
    return APIResponse.success(data=result)

@protected_router.post("/ec-connectivity-check", tags=["EC Verify API's"])
@limiter.limit("1/5 seconds")
@limiter.limit("10/minute")
async def ec_check(
    request: Request,
    baseUrl: Optional[str] = Query("https://prportal.nidw.gov.bd", description="Base URL of EC Portal (optional)"),
    username: str = Query(..., description="EC Portal Username"),
    password: str = Query(..., description="EC Portal Password")
):
    ec = EcVerification(
        baseUrl=baseUrl,
        username=username, # "MMBL.ABD",
        password=password   # "Mmbplc@1234"
    )
    token = await ec.get_access_token()
    return APIResponse.success(
            data=token
        )

@protected_router.post("/ec-verification", tags=["EC Verify API's"])
@limiter.limit("10/minute")
async def ec_check(
    request: Request,
    token_data: dict = Security(verify_jwt_token),
    request_body: EcVerificationRequest = Body(...),  
):
    if(token_data.get("ec_credentials_valid")):
        ec = EcVerification(
            baseUrl=token_data.get("ec_baseurl"),
            username=token_data.get("ec_username"), # "MMBL.ABD",
            password=token_data.get("ec_password"),   # "Mmbplc@1234"
        )
        verification_result = await ec.verify_voter(request_body)
        if verification_result["status"] == 200:
            return APIResponse.success(
                    data = verification_result["result"]
                )
        else:
            return APIResponse.error(
                message="EC checking error."
            ) 
    else:
       return APIResponse.error(
                message="EC credentials are invalid"
            ) 
    
@protected_router.post("/stats", tags=["Hit Counter API's"])
async def get_stats():
    return await hit_counter.get_summary()

@protected_router.get("/history/{requestId}", tags=["History API's"])
async def download_folder(requestId: str):
    folder_path = os.path.join("history", requestId)
    
    if not os.path.isdir(folder_path):
        return APIResponse.error(
                message="RequestID not found"
            )

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename={requestId}.zip"
    })

app.include_router(protected_router)