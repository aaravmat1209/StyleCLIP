import base64
import os
import uuid
import torch
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import glob
from datetime import datetime
from fastapi import HTTPException
from backend.app.schemas.clothing_schemas import (
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

async def process_csv_data():
    """Process all CSV files and populate database with embeddings"""
    data_dir = "/Users/etloaner/Desktop/styleCLIP/StyleCLIP/data"
    csv_files = glob.glob(os.path.join(data_dir, "*_products.csv"))
    all_items = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        brand = os.path.basename(csv_file).replace('_products.csv', '')
        
        for _, row in df.iterrows():
            try:
                # Load image from URL
                response = requests.get(row['image_url'], timeout=10)
                image = Image.open(BytesIO(response.content))
                
                # Get embedding
                embedding = clip_model.get_image_embedding_from_pil(image)
                
                # Extract tags
                garment_type = tag_extractor.determine_garment_type(embedding)
                if garment_type != "Unknown":
                    tags_dict = tag_extractor.extract_tags(embedding, garment_type)
                    tags = list(tags_dict.values())
                else:
                    tags = ["Unknown garment type"]
                
                item = {
                    "product_id": row['product_id'],
                    "name": row['name'],
                    "current_price": row['current_price'],
                    "image_url": row['image_url'],
                    "embedding": embedding.squeeze().tolist(),
                    "tags": tags,
                    "garment_type": garment_type,
                    "brand": brand
                }
                all_items.append(item)
                
            except Exception as e:
                print(f"Error processing {row.get('name', 'unknown')}: {e}")
                continue
    
    # Save to MongoDB
    if all_items:
        db = await get_db().__anext__()
        await db.clothing_items.delete_many({})
        await db.clothing_items.insert_many(all_items)
    
    return {"processed": len(all_items), "message": "CSV data processed successfully"}

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

async def get_similar_items_from_csv(image_url: str = None, limit: int = 5):
    """Get similar items from CSV data based on uploaded image"""
    try:
        if image_url:
            # Process uploaded image
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content))
            query_embedding = clip_model.get_image_embedding_from_pil(image)
        else:
            raise HTTPException(status_code=400, detail="Image URL required")
        
        # Find similar items
        similar_items = await find_similar_clothing(query_embedding, limit)
        
        recommendations = []
        for item, similarity in similar_items:
            recommendations.append({
                "product_id": item['product_id'],
                "name": item['name'],
                "current_price": item['current_price'],
                "image_url": item['image_url'],
                "tags": item['tags'],
                "similarity": f"{similarity * 100:.1f}%",
                "brand": item['brand']
            })
        
        return {"similar_items": recommendations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")

async def get_similar_items(product_id: str, limit: int = 5):
    """Get similar items for a given product ID"""
    try:
        db = await get_db().__anext__()
        
        # Get the original item
        original_item = await db.clothing_items.find_one({"product_id": product_id})
        if not original_item or 'embedding' not in original_item:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Find similar items
        embedding = torch.tensor(original_item['embedding'])
        similar_items = await find_similar_clothing(embedding, limit + 1)
        
        recommendations = []
        for item, similarity in similar_items:
            if item['product_id'] != product_id:
                recommendations.append({
                    "product_id": item['product_id'],
                    "name": item['name'],
                    "current_price": item['current_price'],
                    "image_url": item['image_url'],
                    "tags": item['tags'],
                    "similarity": f"{similarity * 100:.1f}%",
                    "brand": item['brand']
                })
        
        return recommendations[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")