# Mental Health Chatbot - Frontend

React frontend for the Mental Health Prediction Chatbot.

## Features

- **Modern Chat Interface**: Clean, intuitive design with smooth animations
- **Real-time Emotion Detection**: Displays predicted emotions with color-coded badges
- **Grok AI Integration**: Optional API key configuration for AI-powered responses
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Empathetic UI**: Designed with mental health support in mind

## Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Start the development server**:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Configuration

### Grok API Key (Optional)
- Click the ⚙️ icon in the header
- Enter your Grok API key from https://console.x.ai/
- The key is stored in memory only (not persisted)

If no API key is provided, the chatbot will use empathetic fallback responses.

## Building for Production

```bash
npm run build
```

The optimized build will be in the `build/` folder.

## Environment Variables

Create a `.env` file if you want to change the API URL:

```
REACT_APP_API_URL=http://localhost:8000
```

## Features Explained

### Emotion Detection
Every message sent by the user is analyzed by the ML model to predict the emotional state:
- Depression 😔
- Anxiety 😰
- Stress 😫
- Anger 😠
- Happiness 😊
- Sadness 😢
- Fear 😨
- Neutral 😐

### Chat Experience
- Smooth animations for incoming messages
- Typing indicators when bot is responding
- Timestamp for each message
- Auto-scroll to latest message
- Color-coded emotion badges

## Tech Stack

- React 18
- Axios for API calls
- CSS3 with animations
- Responsive design (mobile-first)
