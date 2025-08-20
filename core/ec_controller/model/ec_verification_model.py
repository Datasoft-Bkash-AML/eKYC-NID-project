from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import requests
import base64

class Identify(BaseModel):
    nationalId: Optional[str] = None
    # nid10Digit: Optional[str] = None
    # nid17Digit: Optional[str] = None
    # dateOfBirth: Optional[str] = None

class Address(BaseModel):
    division: Optional[str] = None
    district: Optional[str] = None

class Verify(BaseModel):
    name: Optional[str] = None
    nameEn: Optional[str] = None
    dateOfBirth: Optional[str] = None
    father: Optional[str] = None
    mother: Optional[str] = None
    # spouse: Optional[str] = None
    # nationalId: Optional[str] = None
    # presentAddress: Address = None
    # permanentAddress: Address = None

class EcVerificationRequest(BaseModel):
    identify: Identify
    verify: Verify

def filtered_request_body(request: EcVerificationRequest) -> dict:
    params = {}

    # For the 'identify' part
    # identify_params = {key: value for key, value in request.identify.dict().items() if value not in (None, "")}
    
    # Extract raw NID and date of birth
    nid_number = request.identify.nationalId
    dob = request.verify.dateOfBirth

    identify_params = {}

    # Format NID based on length
    if nid_number and nid_number.isdigit():
        if len(nid_number) == 10:
            identify_params["nid10Digit"] = nid_number
        elif len(nid_number) == 13 and dob:
            try:
                year = datetime.strptime(dob, "%Y-%m-%d").year
                identify_params["nid17Digit"] = f"{year}{nid_number}"
            except ValueError:
                raise ValueError("Invalid date format. Expected YYYY-MM-DD.")
        elif len(nid_number) == 17:
            identify_params["nid17Digit"] = nid_number

    params["identify"] = identify_params
    
    
    # For the 'verify' part
    # verify_params = {key: value for key, value in request.verify.dict().items() if value not in (None, "")}
    # verify_params = {
    #     key: (value.capitalize() if key == "name" and isinstance(value, str) else value)
    #     for key, value in request.verify.dict().items()
    #     if value not in (None, "")
    # }
    acceptable_data = ['dateOfBirth', 'nameEn']
    verify_data = request.verify.dict()
    # Check if all required keys exist and are not empty
    if not all(verify_data.get(key) not in (None, "") for key in acceptable_data):
        return None

    verify_params = {
        key: (
            value.capitalize() if key == "name" and isinstance(value, str) else value
        )
        for key, value in verify_data.items()
        if key in acceptable_data and value not in (None, "")
    }
    
    # # Flatten the 'presentAddress' and 'permanentAddress' into the top level, only if they have values
    # for address_field in [request.verify.presentAddress, request.verify.permanentAddress]:
    #     for addr_key, addr_value in address_field.dict().items():
    #         if addr_value not in (None, ""):
    #             verify_params[addr_key] = addr_value

    # Add 'identify' and 'verify' to the final params only if they contain values
    if identify_params:
        params["identify"] = identify_params
    if verify_params:
        params["verify"] = verify_params

    return params

def build_verification_result(request: EcVerificationRequest, response_data: dict):
    field_results = response_data.get("fieldVerificationResult", {})
    field_data = response_data.get("success", {}).get("data", {})
    verify_data = request.verify.dict()

    face_data = image_url_to_base64(field_data["photo"])

    params = {}
    result = {}

    for field, verification_passed in field_results.items():
        if field in verify_data and verify_data[field] not in (None, ""):
            # params[field] = verify_data[field]
            result[field] = verification_passed

    params = verify_data
    result["photo"] = face_data

    return {
        "nid_data": params,
        "ec_data": result
    }

def image_url_to_base64(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"Failed to fetch image, status code: {response.status_code}")


    
class ApiAuthRequest(BaseModel):
    apiKey: str
    baseUrl: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
