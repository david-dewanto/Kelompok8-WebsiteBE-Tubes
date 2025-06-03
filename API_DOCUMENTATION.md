# Antibiotic Resistance Prediction API Documentation

## Base URL
```
https://your-railway-domain.railway.app
```

## Endpoints

### 1. Root Endpoint
**GET** `/`

Returns basic API information.

**Response:**
```json
{
  "message": "Antibiotic Resistance Prediction API",
  "status": "active"
}
```

---

### 2. Predict Antibiotic Resistance
**POST** `/predict`

Predicts antibiotic resistance/susceptibility for a given epitope sequence.

**Request Body:**
```json
{
  "epitope_sequence": "ESSALAAAQAMASAAAFETA"
}
```

**Parameters:**
- `epitope_sequence` (string, required): The epitope sequence to analyze. Can be any length, will be automatically converted to uppercase.

**Response:**
```json
{
  "predictions": {
    "amikacin": "Susceptible",
    "amoxicillin": "Susceptible",
    "capreomycin": "Susceptible",
    "ciprofloxacin": "Susceptible",
    "ethambutol": "Susceptible",
    "isoniazid": "Susceptible",
    "kanamycin": "Susceptible",
    "moxifloxacin": "Susceptible",
    "pyrazinamide": "Susceptible",
    "rifampin": "Resistant",
    "streptomycin": "Susceptible"
  }
}
```

**Possible Values:**
- Each antibiotic will have either `"Resistant"` or `"Susceptible"` as its value

**Error Responses:**
- **503 Service Unavailable**: Model not loaded
  ```json
  {
    "detail": "Model not loaded"
  }
  ```
- **500 Internal Server Error**: Prediction error
  ```json
  {
    "detail": "Prediction error: [error message]"
  }
  ```

---

### 3. Health Check
**GET** `/health`

Checks if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## CORS Configuration

The API is configured to accept requests from:
- `https://predictresistantibiotics.site`
- `http://localhost:3000`

All methods and headers are allowed.

---

## Example Usage (JavaScript/Fetch)

```javascript
async function predictResistance(sequence) {
  const response = await fetch('https://your-railway-domain.railway.app/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      epitope_sequence: sequence
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.predictions;
}

// Example usage
try {
  const predictions = await predictResistance("ESSALAAAQAMASAAAFETA");
  console.log(predictions);
} catch (error) {
  console.error('Error:', error);
}
```

---

## Example Usage (React)

```jsx
import { useState } from 'react';

function AntibioticPrediction() {
  const [sequence, setSequence] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('https://your-railway-domain.railway.app/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          epitope_sequence: sequence
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={sequence}
        onChange={(e) => setSequence(e.target.value)}
        placeholder="Enter epitope sequence"
      />
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      
      {error && <div>Error: {error}</div>}
      
      {predictions && (
        <div>
          <h3>Predictions:</h3>
          {Object.entries(predictions).map(([antibiotic, status]) => (
            <div key={antibiotic}>
              {antibiotic}: <span className={status === 'Resistant' ? 'resistant' : 'susceptible'}>
                {status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## Notes

1. The epitope sequence is automatically converted to uppercase and 'N' characters are removed during processing.
2. The model uses k-mer featurization with k=6 to analyze the sequences.
3. The model predicts resistance/susceptibility for 11 different antibiotics simultaneously.
4. Empty or invalid sequences may result in prediction errors.

---

## Rate Limiting

Currently, there are no rate limits implemented. Consider implementing rate limiting in production if needed.