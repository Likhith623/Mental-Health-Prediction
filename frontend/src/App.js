import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: "Hello! I'm here to listen and support you. How are you feeling today?",
      emotion: null,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getEmotionColor = (emotion) => {
    const colors = {
      'depression': '#6c5ce7',
      'anxiety': '#fd79a8',
      'stress': '#fdcb6e',
      'anger': '#d63031',
      'happiness': '#00b894',
      'sadness': '#74b9ff',
      'fear': '#a29bfe',
      'neutral': '#636e72'
    };
    return colors[emotion?.toLowerCase()] || '#636e72';
  };

  const getEmotionEmoji = (emotion) => {
    const emojis = {
      'depression': '😔',
      'anxiety': '😰',
      'stress': '😫',
      'anger': '😠',
      'happiness': '😊',
      'sadness': '😢',
      'fear': '😨',
      'neutral': '😐'
    };
    return emojis[emotion?.toLowerCase()] || '💭';
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      text: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        message: inputMessage,
        gemini_api_key: apiKey || null
      });

      const botMessage = {
        type: 'bot',
        text: response.data.bot_response,
        emotion: response.data.predicted_emotion,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        type: 'bot',
        text: "I'm sorry, I'm having trouble connecting right now. Please try again.",
        emotion: 'neutral',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <h1>💚 Mental Health Support Chat</h1>
            <p>A safe space to share your feelings</p>
          </div>
          <button 
            className="api-key-toggle"
            onClick={() => setShowApiKeyInput(!showApiKeyInput)}
            title="Configure Gemini API Key"
          >
            ⚙️
          </button>
        </div>

        {showApiKeyInput && (
          <div className="api-key-container">
            <input
              type="password"
              placeholder="Enter Gemini API Key (optional)"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="api-key-input"
            />
            <small>Get your API key from <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a></small>
          </div>
        )}

        <div className="messages-container">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.type}`}>
              <div className="message-content">
                <p>{msg.text}</p>
                {msg.emotion && (
                  <div 
                    className="emotion-badge"
                    style={{ backgroundColor: getEmotionColor(msg.emotion) }}
                  >
                    {getEmotionEmoji(msg.emotion)} {msg.emotion}
                  </div>
                )}
              </div>
              <span className="message-time">
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Share what's on your mind..."
            rows="1"
            disabled={isLoading}
          />
          <button 
            onClick={sendMessage} 
            disabled={!inputMessage.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </div>

        <div className="disclaimer">
          <small>
            💡 This is a supportive tool, not a replacement for professional mental health care.
            {!apiKey && ' Configure Gemini API key for AI-powered responses.'}
          </small>
        </div>
      </div>
    </div>
  );
}

export default App;
