// src/App.jsx
import { useState } from 'react';
import axios from 'axios';
import './App.css'; // We will create this file next
import Latex from 'react-latex-next';
// Your FastAPI backend URL
const API_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // --- Main function to send a question to the /ask endpoint ---
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: input,
      sender: 'user',
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Call your FastAPI /ask endpoint
      const response = await axios.post(`${API_URL}/ask`, {
        question: input,
      });

      const { answer, sources, interaction_id } = response.data;

      const aiMessage = {
        id: interaction_id, // Use the interaction_id as the message ID
        text: answer,
        sender: 'ai',
        sources: sources,
        // Store a copy of the original question for the feedback payload
        originalQuestion: input, 
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error fetching answer:', error);
      const errorMessage = {
        id: Date.now(),
        text: 'Sorry, something went wrong. Please try again.',
        sender: 'ai',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // --- HITL function to send feedback to the /feedback endpoint ---
  const handleFeedback = async (message, isGood) => {
    let correctedAnswer = null;

    // If feedback is bad, ask for the correct answer (as per assignment)
    if (!isGood) {
      correctedAnswer = prompt(
        "Sorry about that! What was the correct answer?"
      );
    }

    try {
      await axios.post(`${API_URL}/feedback`, {
        interaction_id: message.id,
        question: message.originalQuestion,
        answer: message.text,
        is_good: isGood,
        corrected_answer: correctedAnswer,
        sources: message.sources,
      });

      // Update the message state to show feedback was sent
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === message.id
            ? { ...msg, feedbackSent: true, isGood: isGood }
            : msg
        )
      );
    } catch (error) {
      console.error('Error sending feedback:', error);
      alert('Failed to send feedback.');
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Math Professor Agent 🧠</h1>
      </div>
      <div className="chat-window">
        {messages.map((msg) => (
          <Message key={msg.id} message={msg} onFeedback={handleFeedback} />
        ))}
        {isLoading && <div className="message ai-message">Thinking...</div>}
      </div>
      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a math question..."
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading}>
          Send
        </button>
      </div>
    </div>
  );
}

// --- Message Component (Handles rendering and feedback) ---
function Message({ message, onFeedback }) {
  const isAI = message.sender === 'ai';

  return (
    <div className={`message ${isAI ? 'ai-message' : 'user-message'}`}>
      {/* 2. USE <Latex> for AI messages, <p> for user messages */}
      {isAI ? (
        <Latex>{message.text}</Latex>
      ) : (
        <p>{message.text}</p>
      )}

      {/* Show sources if they exist */}
      {isAI && message.sources && message.sources.length > 0 && (
        <div className="sources">
          <strong>Sources:</strong>
          <ul>
            {message.sources.map((src, index) => (
              <li key={index}>{src.substring(0, 100)}...</li>
            ))}
          </ul>
        </div>
      )}

      {/* Show Feedback buttons for AI messages */}
      {isAI && (
        <div className="feedback-buttons">
          {!message.feedbackSent ? (
            <>
              <button onClick={() => onFeedback(message, true)}>👍</button>
              <button onClick={() => onFeedback(message, false)}>👎</button>
            </>
          ) : (
            <span className="feedback-thanks">
              {message.isGood ? 'Thanks! 👍' : 'Thanks for the feedback! 👎'}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default App;