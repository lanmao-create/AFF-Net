# AFF-Net
Semantic segmentation model for high-resolution remote sensing images
## Visualization

![result](Visualization/FDF.png)
Visual analysis of the weight generation pathway in the proposed frequency-adaptive weighting mechanism. (a) Raw input patches from the VIS and DSM modalities. (b) Feature heatmaps (Following standard heatmap visualization conventions, warmer colors (e.g., dark red) denote strong feature responses). (c) The discrete wavelet transform (DWT) decomposition step, illustrating the low-frequency structure (LL) and high-frequency details (LH, HL, and HH). (d) Visualization of the frequency-guided spatial weights (enhancement strength). [CZ1.1](e) The fused multi-frequency sub-bands obtained by merging the corresponding frequency subbands from both modalities. (f) The final fused feature heatmap reconstructed via the inverse discrete wavelet transform (IDWT).

![result](Visualization/Baseline.png)
The baseline fusion method (standard channel-wise concatenation). (a) Raw VIS and DSM input images. (b) Feature heatmaps (Following standard heatmap visualization conventions, warmer colors (e.g., dark red) denote regions with high activation responses). (c) The final fused feature heatmap obtained by directly concatenating the extracted features along the channel dimension, without frequency decomposition or adaptive spatial weighting.

![result](Visualization/SDA.png)
Visual analysis of the high-frequency detail alignment process in the decoder. (a) Visualization of the target detail map generation. It displays the high-frequency details extracted from the original input image and the residual calculated against the current decoder features, representing the missing spatial details that require supplementation. (b) The refined decoder feature heatmap after the high-frequency detail compensation, the clearly delineated boundary responses demonstrate the successful restoration of spatial structures.
