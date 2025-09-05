import React from 'react';

function Recommendations({ recommendations }) {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 w-full max-w-2xl bg-white p-4 rounded shadow">
      <h2 className="text-xl font-semibold mb-4">Similar Items:</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {recommendations.map((item) => (
          <div key={item.id} className="border rounded p-3">
            <div className="font-medium text-sm mb-2">{item.filename}</div>
            <div className="text-xs text-gray-600 mb-2">
              Similarity: {(item.similarity * 100).toFixed(1)}%
            </div>
            <div className="text-xs">
              <strong>Tags:</strong> {item.tags.join(', ')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Recommendations;
