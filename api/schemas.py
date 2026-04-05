from pydantic import BaseModel
from typing import List, Dict

class RequestData(BaseModel):
    users: List[Dict]
    posts: List[Dict]
    comments: List[Dict]