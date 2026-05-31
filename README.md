A PyTorch implementation of Neural Style Transfer based on the paper 'A Neural Algorithm of Artistic Style' (Gatys et al., 2015).

Uses a pretrained VGG19 network to extract content features from deep layers and style features from early layers via Gram Matrix computation. Optimizes a generated image using gradient descent on pixels — not weights — to minimize a weighted combination of content and style loss.

Features:
- GPU acceleration via CUDA
- Adjustable content/style weight balance (α/β)
- Configurable image size and optimization steps
- Save output to disk

Tech stack: PyTorch, torchvision, PIL, NumPy, Matplotlib
