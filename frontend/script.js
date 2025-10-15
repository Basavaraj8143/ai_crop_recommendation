document.getElementById('cropForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;
    submitButton.innerHTML = 'Loading...';
    submitButton.disabled = true;

    // Validate inputs
    const inputs = ['n', 'p', 'k', 'temp', 'humidity', 'ph', 'rainfall'];
    const missingInputs = inputs.filter(id => !document.getElementById(id).value);
    
    if (missingInputs.length > 0) {
        alert(`Please fill in all required fields: ${missingInputs.join(', ')}`);
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
        return;
    }

    // Gather input values
    const data = {
        n: parseFloat(document.getElementById('n').value),
        p: parseFloat(document.getElementById('p').value),
        k: parseFloat(document.getElementById('k').value),
        temperature: parseFloat(document.getElementById('temp').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };

    try {
        console.log('Sending data:', data);  // Debug log
        
        const response = await fetch('http://127.0.0.1:5000/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Received result:', result);  // Debug log

        // Show the result container
        const resultDiv = document.getElementById('result');
        resultDiv.classList.remove('hidden');

        // Recommended crop
        document.getElementById('recommendedCrop').innerText = result.recommended_crop;

        // Top 3 crops
        const top3Div = document.getElementById('top3Crops');
        top3Div.innerHTML = ''; // Clear previous
        result.top_3_crops.forEach((crop, index) => {
            const p = document.createElement('p');
            p.innerText = `${index + 1}. ${crop}`;
            p.className = 'text-green-800 font-semibold';
            top3Div.appendChild(p);
        });

        // Create chart
        const ctx = document.getElementById('top3Chart').getContext('2d');
        if (window.top3ChartInstance) {
            window.top3ChartInstance.destroy();
        }
        window.top3ChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: result.top_3_crops,
                datasets: [{
                    label: 'Confidence Score',
                    data: result.top_3_crops.map((_, i) => 
                        i === 0 ? result.confidence : (result.confidence * (0.8 - i * 0.2))),
                    backgroundColor: [
                        'rgba(34, 197, 94, 0.8)',  // Green-500
                        'rgba(34, 197, 94, 0.6)',
                        'rgba(34, 197, 94, 0.4)'
                    ],
                    borderColor: [
                        'rgb(21, 128, 61)',  // Green-700
                        'rgb(21, 128, 61)',
                        'rgb(21, 128, 61)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Top 3 Crop Recommendations'
                    }
                }
            }
        });

        // Confidence
        document.getElementById('confidence').innerText = `Confidence: ${result.confidence}%`;

    } catch (error) {
        console.error('Error:', error);  // Debug log
        alert("Error: " + error.message);
    } finally {
        // Reset button state
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
    }
});
