"""
Web-based Conversational AI Interface for Fraud Detection
Flask web app with chat interface
"""
from flask import Flask, render_template, request, jsonify
from conversational_ai import FraudDetectionChatbot
import threading
import webbrowser
import time

app = Flask(__name__)
chatbot = None

def initialize_chatbot():
    """Initialize the chatbot with trained models"""
    global chatbot
    print("🔧 Initializing fraud detection models...")
    chatbot = FraudDetectionChatbot()
    
    # Train models in background
    training_data = chatbot.fraud_detector.generate_synthetic_data(n_samples=2000, fraud_ratio=0.02)
    chatbot.fraud_detector.train_models(training_data)
    print("✅ Fraud Detection AI is ready!")

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if not chatbot:
        return jsonify({
            'response': 'System is still initializing. Please wait a moment...',
            'status': 'loading'
        })
    
    user_message = request.json.get('message', '')
    
    if not user_message.strip():
        return jsonify({
            'response': 'Please enter a message.',
            'status': 'error'
        })
    
    try:
        bot_response = chatbot.process_message(user_message)
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'response': f'Sorry, I encountered an error: {str(e)}',
            'status': 'error'
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if chatbot else 'initializing',
        'timestamp': time.time()
    })

# Create templates directory and HTML template
import os

def create_web_interface():
    """Create the HTML template for the web interface"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Fraud Detection AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 5px;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
        }
        
        .user-avatar {
            background: #007bff;
        }
        
        .bot-avatar {
            background: #28a745;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }
        
        .chat-input input:focus {
            border-color: #007bff;
        }
        
        .chat-input button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        .chat-input button:hover {
            background: #0056b3;
        }
        
        .chat-input button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
            font-style: italic;
            color: #666;
        }
        
        .example-queries {
            padding: 10px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
        }
        
        .example-queries strong {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            🤖 Fraud Detection AI Assistant
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message bot">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your AI fraud detection assistant. I can help you analyze transactions, explain fraud decisions, and answer questions about our system.

Try asking me something like:
• "Check transaction: $1500, online purchase, 3am"
• "Why was this transaction flagged?"
• "Show system performance"
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typing">
            🤖 AI is thinking...
        </div>
        
        <div class="example-queries">
            <strong>Example queries:</strong> "Check $500 gas purchase at 2am" | "Explain fraud decision" | "System stats" | "Help"
        </div>
        
        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Ask me about fraud detection..." autocomplete="off">
            <button onclick="sendMessage()" id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typing');

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = `message-avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`;
            avatar.textContent = isUser ? 'You' : '🤖';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            if (isUser) {
                messageDiv.appendChild(contentDiv);
                messageDiv.appendChild(avatar);
            } else {
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(contentDiv);
            }
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTyping() {
            typingIndicator.style.display = 'block';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.style.display = 'none';
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            
            // Disable input while processing
            sendButton.disabled = true;
            messageInput.disabled = true;
            showTyping();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                hideTyping();
                
                // Add bot response
                addMessage(data.response);
                
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.');
            } finally {
                // Re-enable input
                sendButton.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }

        // Handle Enter key
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Focus input on load
        messageInput.focus();
        
        // Check system status
        setInterval(async () => {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                // Could update UI based on system status
            } catch (error) {
                console.log('Health check failed');
            }
        }, 30000);
    </script>
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'chat.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def run_web_app():
    """Run the Flask web application"""
    # Create web interface template
    create_web_interface()
    
    # Initialize chatbot in background
    init_thread = threading.Thread(target=initialize_chatbot)
    init_thread.daemon = True
    init_thread.start()
    
    print("🚀 Starting Fraud Detection AI Web Interface...")
    print("🌐 Open your browser to: http://localhost:5000")
    
    # Auto-open browser after a delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Flask app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    run_web_app()
