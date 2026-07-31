import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the default board action', () => {
  render(<App />);
  expect(screen.getByRole('button', { name: /play default board/i })).toBeInTheDocument();
});
