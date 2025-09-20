import logging
from fastapi import APIRouter, HTTPException
from backend.app.schemas.clothing_schemas import (
    TagRequest,
    TagResponse
)
from backend.app.controllers.clothing_controller import (
    handle_tag_request,
    get_similar_items_from_csv,
    process_csv_data
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clothing", tags=["Clothing"])

@router.post("/process-csv")
async def process_csv_data_endpoint():
    """
    Process CSV files and generate embeddings for all products.
    """
    try:
        return await process_csv_data()
    except Exception as e:
        logger.exception("Error processing CSV data")
        raise HTTPException(status_code=500, detail="Failed to process CSV data.")

@router.post("/tag", response_model=TagResponse)
async def tag_clothing_image(payload: TagRequest):
    """
    Tag a clothing image with garment type and relevant features without storing the item.
    """
    try:
        return await handle_tag_request(payload)
    except Exception as e:
        logger.exception("Error during clothing image tagging")
        raise HTTPException(status_code=500, detail="Failed to extract tags from image.")

@router.get("/recommendations")
async def get_recommendations(image_url: str = None, limit: int = 5):
    """
    Get clothing recommendations based on uploaded image or similar items from CSV data.
    """
    try:
        return await get_similar_items_from_csv(image_url, limit)
    except Exception as e:
        logger.exception("Error getting recommendations")
        raise HTTPException(status_code=500, detail="Failed to get recommendations.")

@router.get("/similar/{product_id}")
async def get_similar_clothing(product_id: str, limit: int = 5):
    """
    Get similar clothing items based on product ID from CSV data
    """
    try:
        return await get_similar_items(product_id, limit)
    except Exception as e:
        logger.exception("Error getting similar items")
        raise HTTPException(status_code=500, detail="Failed to find similar items.")
