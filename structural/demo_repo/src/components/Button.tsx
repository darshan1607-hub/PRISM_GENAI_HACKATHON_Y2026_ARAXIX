import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({ label, onClick, disabled = false }) => {
  return (
    <button className="primary-btn" onClick={onClick} disabled={disabled}>
      <span>{label}</span>
    </button>
  );
};
