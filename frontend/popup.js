
// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('summarizeButton').addEventListener('click', function() {
//         const url = document.getElementById('urlInput').value;
  
//         fetch('http://localhost:5000/summarize-url', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({ url: url })
//         })
//         .then(response => response.json())
//         .then(data => {
//             document.getElementById('summaryOutput').innerText = data.summary;
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     });
//   });

document.getElementById('summarize-button').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      let activeTab = tabs[0];
      let activeTabUrl = activeTab.url;
  
      fetch('http://127.0.0.1:5000/summarize-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: activeTabUrl })
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById('summary').innerText = data.summary;
      })
      .catch((error) => {
        console.error('Error:', error);
        document.getElementById('summary').innerText = 'Error summarizing the privacy policy.';
      });
    });
  });
  
  
