// chrome.runtime.onInstalled.addListener(() => {
//     console.log('Text Summarizer extension installed');
//   });
  
chrome.action.onClicked.addListener((tab) => {
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
      console.log('Summary:', data.summary);
    })
    .catch((error) => {
      console.error('Error:', error);
    });
  });
});
