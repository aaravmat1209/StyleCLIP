import { useState } from 'react';
import { uploadClothingItem, tagClothingImage, getSimilarItems } from './utils/api';
import ImageUpload from './components/ImageUpload';
import DetectedTags from './components/DetectedTags';
import Recommendations from './components/Recommendations';

function App() {
  const [image, setImage] = useState(null);
  const [detectedItems, setDetectedItems] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    try {
      // Step 1: Upload the clothing item to the backend
      const uploadResponse = await uploadClothingItem(image);
      setDetectedItems(uploadResponse.tags || []);

      // Step 2: Get similar items using the uploaded item's ID
      if (uploadResponse.id) {
        const similarItems = await getSimilarItems(uploadResponse.id);
        setRecommendations(similarItems || []);
      }
    } catch (error) {
      console.error('Error uploading or getting similar items:', error);
      setError("Something went wrong while processing the image. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <h1 className="text-2xl font-bold mb-4">Clothing Recommender</h1>

      <ImageUpload setImage={setImage} />

      {/* Add a button for uploading the image */}
      <button 
        onClick={handleUpload} 
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Upload and Get Recommendations
      </button>

      {error && (
        <div className="mt-6 text-red-600 font-semibold">
          <p>{error}</p>
        </div>
      )}

      {detectedItems.length > 0 && (
        <DetectedTags detectedItems={detectedItems} />
      )}

      {recommendations.length > 0 && (
        <Recommendations recommendations={recommendations} />
      )}
    </div>
  );
}

export default App;
