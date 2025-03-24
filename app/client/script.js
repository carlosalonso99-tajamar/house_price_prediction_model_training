document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    // Get form data
    const formData = {
        area: document.getElementById('area').value,
        bedrooms: document.getElementById('bedrooms').value,
        bathrooms: document.getElementById('bathrooms').value,
        stories: document.getElementById('stories').value,
        mainroad: document.getElementById('mainroad').value,
        guestroom: document.getElementById('guestroom').value,
        basement: document.getElementById('basement').value,
        hotwaterheating: document.getElementById('hotwaterheating').value,
        airconditioning: document.getElementById('airconditioning').value,
        parking: document.getElementById('parking').value,
        prefarea: document.getElementById('prefarea').value,
        furnishingstatus: document.getElementById('furnishingstatus').value
    };
    
    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Format the price with commas
            const formattedPrice = new Intl.NumberFormat('en-IN').format(data.predicted_price);
            document.getElementById('predictedPrice').textContent = `₹${formattedPrice}`;
            document.getElementById('result').style.display = 'block';
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        document.getElementById('error').textContent = error.message;
        document.getElementById('error').style.display = 'block';
    }
}); 