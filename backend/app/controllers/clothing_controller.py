import base64
import os
import uuid
import torch
from datetime import datetime
from fastapi import HTTPException
from backend.app.schemas.clothing_schemas import (
    UploadClothingItemRequest,
    UploadClothingItemResponse,
    TagRequest,
    TagResponse
)
from backend.app.models.clip_model import CLIPModel
from backend.app.controllers.tag_extractor import TagExtractor
from backend.app.config.tag_list_en import GARMENT_TYPES
from backend.app.config.database import get_db  

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

clip_model = CLIPModel()
tag_extractor = TagExtractor(tag_dict=GARMENT_TYPES)

async def find_similar_clothing(embedding: torch.Tensor, limit: int = 5):
    """Find similar clothing items using cosine similarity"""
    db = await get_db().__anext__()
    
    # Get all clothing items from database
    cursor = db.clothing_items.find({})
    similar_items = []
    
    async for item in cursor:
        if 'embedding' in item:
            stored_embedding = torch.tensor(item['embedding'])
            similarity = torch.cosine_similarity(embedding, stored_embedding, dim=0).item()
            similar_items.append((item, similarity))
    
    # Sort by similarity and return top items
    similar_items.sort(key=lambda x: x[1], reverse=True)
    return similar_items[:limit]

async def handle_upload_clothing_item(payload: UploadClothingItemRequest) -> UploadClothingItemResponse:
    image_data = base64.b64decode(payload.image_base64)
    filename = payload.filename or f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        f.write(image_data)

    try:
        # Step 1: Get embedding
        embedding = clip_model.get_image_embedding(image_path)

        # Step 2: Garment Type Classification
        garment_type = tag_extractor.determine_garment_type(embedding)

        # Step 3: Feature Tag Extraction
        if garment_type != "Unknown":
            tags_dict = tag_extractor.extract_tags(embedding, garment_type)
            tags = list(tags_dict.values())
        else:
            tags = ["Unknown garment type"]

        # Step 4: Save to MongoDB
        db = await get_db().__anext__()
        clothing_item = {
            "filename": filename,
            "image_path": image_path,
            "embedding": embedding.squeeze().tolist(),
            "tags": tags,
            "garment_type": garment_type,
            "uploaded_at": datetime.utcnow()
        }
        result = await db.clothing_items.insert_one(clothing_item)
        item_id = str(result.inserted_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return UploadClothingItemResponse(
        id=item_id,
        filename=filename,
        tags=tags
    )

async def handle_tag_request(payload: TagRequest) -> TagResponse:
    image_data = base64.b64decode(payload.image_base64)
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(image_path, "wb") as f:
        f.write(image_data)

    try:
        embedding = clip_model.get_image_embedding(image_path)
        garment_type = tag_extractor.determine_garment_type(embedding)
        if garment_type != "Unknown":
            tags_dict = tag_extractor.extract_tags(embedding, garment_type)
            tags = list(tags_dict.values())
            print(f"Debug TAG endpoint - Tags being returned: {tags}")
        else:
            tags = ["Unknown garment type"]
            print(f"Debug TAG endpoint - Unknown garment, returning: {tags}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return TagResponse(tags=tags)

async def get_similar_items(item_id: str, limit: int = 5):
    """Get similar items for a given clothing item"""
    try:
        from bson import ObjectId
        db = await get_db().__anext__()
        
        # Get the original item
        original_item = await db.clothing_items.find_one({"_id": ObjectId(item_id)})
        if not original_item or 'embedding' not in original_item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Find similar items
        embedding = torch.tensor(original_item['embedding'])
        similar_items = await find_similar_clothing(embedding, limit + 1)  # +1 to exclude self
        
        # Filter out the original item and format response
        recommendations = []
        for item, similarity in similar_items:
            if str(item['_id']) != item_id:  # Exclude the original item
                recommendations.append({
                    "id": str(item['_id']),
                    "filename": item['filename'],
                    "tags": item['tags'],
                    "similarity": round(similarity, 3)
                })
        
        return recommendations[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")