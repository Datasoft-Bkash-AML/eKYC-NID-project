from fastapi.responses import JSONResponse
from typing import Optional, Any


class APIResponse:
    @staticmethod
    def success(message: str = "Request successful", data: Optional[Any] = None) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": message,
                "data": data,
                "error": None
            }
        )

    @staticmethod
    def error(message: str = "Something went wrong", code: int = 400, details: Optional[str] = None) -> JSONResponse:
        return JSONResponse(
            status_code=code,
            content={
                "success": False,
                "message": message,
                "data": None,
                "error": {
                    "code": code,
                    "details": details
                }
            }
        )
