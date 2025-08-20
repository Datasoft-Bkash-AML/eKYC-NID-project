from jose import JWTError, jwt
from typing import Dict
import os
from fastapi import Body, Depends, FastAPI, UploadFile, File, HTTPException, Security, Header, Request, APIRouter, Form, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

API_KEY = os.getenv("API_KEY", "ds") 
API_KEY_NAME = "X-API-Key"

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_token(data: Dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# Decode JWT token and verify
def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token"
        )