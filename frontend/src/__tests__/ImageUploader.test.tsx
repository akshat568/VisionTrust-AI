import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ImageUploader } from '../components/ImageUploader';

describe('ImageUploader Component', () => {
  it('renders upload area and analyze button disabled initially', () => {
    render(
      <ImageUploader
        onAnalyze={vi.fn()}
        isLoading={false}
        onReset={vi.fn()}
        selectedFile={null}
        setSelectedFile={vi.fn()}
        previewUrl={null}
        setPreviewUrl={vi.fn()}
      />
    );

    expect(screen.getByText(/Drag & drop an image here/i)).toBeInTheDocument();
    const button = screen.getByRole('button', { name: /Analyze Image/i });
    expect(button).toBeDisabled();
  });

  it('renders selected file metadata and enables analyze button', () => {
    const dummyFile = new File(['dummy content'], 'test_cat.png', { type: 'image/png' });

    render(
      <ImageUploader
        onAnalyze={vi.fn()}
        isLoading={false}
        onReset={vi.fn()}
        selectedFile={dummyFile}
        setSelectedFile={vi.fn()}
        previewUrl="blob:http://localhost/test"
        setPreviewUrl={vi.fn()}
      />
    );

    expect(screen.getByText('test_cat.png')).toBeInTheDocument();
    const button = screen.getByRole('button', { name: /Analyze Image/i });
    expect(button).not.toBeDisabled();
  });
});
