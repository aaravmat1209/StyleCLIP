import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
import asyncio
from backend.app.models.clip_model import CLIPModel
from backend.app.controllers.tag_extractor import TagExtractor
from backend.app.config.tag_list_en import GARMENT_TYPES
from backend.app.config.database import get_db
import os
import glob

class CSVDataProcessor:
    def __init__(self):
        self.clip_model = CLIPModel()
        self.tag_extractor = TagExtractor(tag_dict=GARMENT_TYPES)
        
    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL with headers to avoid blocking"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None
    
    def process_csv_file(self, csv_path: str) -> list:
        """Process a single CSV file and return product data with embeddings"""
        df = pd.read_csv(csv_path)
        processed_items = []
        
        for _, row in df.iterrows():
            try:
                # Load image from URL
                image = self.load_image_from_url(row['image_url'])
                if image is None:
                    continue
                
                # Get embedding
                embedding = self.clip_model.get_image_embedding_from_pil(image)
                
                # Extract tags
                garment_type = self.tag_extractor.determine_garment_type(embedding)
                if garment_type != "Unknown":
                    tags_dict = self.tag_extractor.extract_tags(embedding, garment_type)
                    tags = list(tags_dict.values())
                else:
                    tags = ["Unknown garment type"]
                
                # Create product item
                item = {
                    "product_id": row['product_id'],
                    "name": row['name'],
                    "current_price": row['current_price'],
                    "original_price": row['original_price'],
                    "discount": row['discount'],
                    "available_sizes": row['available_sizes'],
                    "colors": row['colors'],
                    "availability": row['availability'],
                    "url": row['url'],
                    "image_url": row['image_url'],
                    "embedding": embedding.squeeze().tolist(),
                    "tags": tags,
                    "garment_type": garment_type,
                    "brand": os.path.basename(csv_path).replace('_products.csv', '')
                }
                processed_items.append(item)
                print(f"Processed: {row['name']}")
                
            except Exception as e:
                print(f"Error processing {row.get('name', 'unknown')}: {e}")
                continue
        
        return processed_items
    
    async def process_all_csv_files(self, data_dir: str):
        """Process working CSV files (skip problematic ones)"""
        # Only process files that work (skip altardstate due to 403 errors)
        working_files = ['edikted_products.csv', 'cupshe_products.csv', 'gymshark_products.csv', 'nakd_products.csv', 'princess_polly.csv', 'vuori_products.csv']
        all_items = []
        
        for filename in working_files:
            csv_file = os.path.join(data_dir, filename)
            if os.path.exists(csv_file):
                print(f"Processing {csv_file}...")
                items = self.process_csv_file(csv_file)
                all_items.extend(items)
                print(f"Processed {len(items)} items from {filename}")
        
        # Save to MongoDB
        if all_items:
            db = await get_db().__anext__()
            await db.clothing_items.delete_many({})  # Clear existing data
            await db.clothing_items.insert_many(all_items)
            print(f"Saved {len(all_items)} items to database")
        
        return all_items

async def main():
    processor = CSVDataProcessor()
    data_dir = "/Users/etloaner/Desktop/styleCLIP/StyleCLIP/data"
    await processor.process_all_csv_files(data_dir)

if __name__ == "__main__":
    asyncio.run(main())