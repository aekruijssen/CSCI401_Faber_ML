from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def sample(self, n_samples):
        reference_image = null
        target_images = null
        label = null
        return reference_image, target_images, label

    @abstractmethod
    def sample_batch(self, batch_size, n_samples):
        reference_image_np = null
        target_images_np = null
        labels_np = null
        return reference_image_np, target_images_np, labels_np