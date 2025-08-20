import httpx
import time
import asyncio

from core.ec_controller.model.ec_verification_model import EcVerificationRequest, build_verification_result, filtered_request_body
from utils.logger import log_json

class EcVerification:
    def __init__(self, baseUrl: str, username: str, password: str):
        self.baseUrl = baseUrl  # e.g. "https://prportal.nidw.gov.bd"
        self.username = username
        self.password = password
        self.token_cache = {
            "access_token": None,
            "refresh_token": None,
            "expires_at": 0  # UNIX timestamp
        }

    async def login(self):
        async with httpx.AsyncClient() as client:
            payload = {
                "username": self.username,
                "password": self.password
            }

            try:
                response = await client.post(
                    f"{self.baseUrl}/partner-service/rest/auth/login",
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    tokens = data["success"]["data"]
                    self.token_cache["access_token"] = tokens["access_token"]
                    self.token_cache["refresh_token"] = tokens["refresh_token"]
                    self.token_cache["expires_at"] = time.time() + 900  # or use expires_in
                    return {
                        "status" : 200,
                        "access_token" : tokens["access_token"]
                    }

                elif response.status_code == 503:
                    # print("🚫 Service unavailable or permission issue (503).")
                    return {
                        "status" : 503,
                        "message" : f"🚫 Service unavailable or permission issue (503).",
                    }

                else:
                    # print(f"⚠️ Unexpected status: {response.status_code}")
                    print("Response:", response.text)
                    return {
                        "status" : 400,
                        "message" : f"⚠️ Unexpected status: {response.status_code}",
                    }

            except httpx.RequestError as e:
                # print(f"❌ Network error while connecting to {self.baseUrl}:\n{e}")
                return {
                    "status" : 400,
                    "message" : "❌ Network error while connecting to {self.baseUrl}:\n{e}",
                }

            except Exception as e:
                # print(f"❗ Unexpected error during login:\n{e}")
                return {
                    "status" : 400,
                    "message" : "❗ Unexpected error during login:\n{e}"
                }

    async def refresh(self):
        async with httpx.AsyncClient() as client:
            payload = {
                "refresh_token": self.token_cache["refresh_token"]
            }
            response = await client.post(
                f"{self.baseUrl}/partner-service/rest/auth/refresh", 
                json=payload
            )
            data = response.json()

            try:
                tokens = data["success"]["data"]
                self.token_cache["access_token"] = tokens["access_token"]
                self.token_cache["refresh_token"] = tokens["refresh_token"]
                self.token_cache["expires_at"] = time.time() + 900
                return {
                        "status" : 200,
                        "token" : tokens["access_token"]
                    }
            
            except (KeyError, TypeError) as e:
                # print("Refresh error:", e)
                return {
                    "status" : 400,
                    "message" : "Refresh error: \n{e}"
                }

    async def get_access_token(self):
        now = time.time()
        if self.token_cache["access_token"] and now < self.token_cache["expires_at"]:
            return self.token_cache["access_token"]
        elif self.token_cache["refresh_token"]:
            # return await self.refresh()
            return await self.login()
        else:
            return await self.login()
        

    async def verify_voter(self, request: EcVerificationRequest):

        # Check if token is expired
        if not self.token_cache["access_token"] or self.token_cache["expires_at"] < time.time():
            await self.login()
        
        if self.token_cache['access_token'] is None:
            return  {
                "status" : 500,
                "mesage" : "EC Token Error"
            }

        headers = {
            "Authorization": f"Bearer {self.token_cache['access_token']}",
            "Content-Type": "application/json"
        }

        # Only include non-empty fields from the dataclass
        params = filtered_request_body(request)

        if params is None:
            return {
                    "status" : 404,
                    "message" : "EC process failed due to data missing !"
                }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.baseUrl}/partner-service/rest/voter/demographic/verification",
                    headers=headers,
                    json=params
                )

                # log_json(f"EC Request", [request, response])

                if response.status_code in (200,406):
                    response_data = response.json()
                    result = build_verification_result(request, response_data)
                    return {
                        "status" : 200,
                        "result" : result
                    }
                
                else:
                    response_data = response.json()
                    print(f"⚠️ Error {response.status_code}: {response.text}")
                    return {
                        "status" : response.status_code,
                        "message" : response_data["error"]["message"]
                    }

            except httpx.RequestError as e:
                print(f"❌ Network error: {e}")
                return {
                    "status" : 500,
                    "mesage" : "EC Network Connectivity Error"
                }


