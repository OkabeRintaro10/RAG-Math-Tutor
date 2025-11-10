import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error: error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error("Error rendering LaTeX:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <div className="latex-error">
          <p>Error: Failed to render mathematical content.</p>
          <pre>{this.state.error.message}</pre>
        </div>
      );
    }

    return this.props.children; 
  }
}

export default ErrorBoundary;