import base64
import io
from PIL import Image, ImageOps


class Progress:
    def __init__(self, shared, steps, total):
        self.shared, self.steps, self.total = shared, steps, total
        self.index = self.completed = 0
        state = shared.state
        state.job_count, state.job_no = total, 0
        state.sampling_steps = steps + 1  # final decoding is real work too
        state.current_latent = None
        self.preview = bool(getattr(shared.opts, 'live_previews_enable', False))
        shared.total_tqdm.updateTotal((steps + 1) * total)

    def status(self, text):
        self.shared.state.textinfo = f'H3: image {self.index+1}/{self.total} | {text}'

    def start(self, index, width, height):
        self.index, self.completed = index, 0
        state = self.shared.state
        state.job_no, state.sampling_step, state.preview_step = index, 0, 0
        state.current_image = state.current_latent = None
        state.current_image_sampling_step = 0
        state.job = f'H3 image {index+1}/{self.total}'
        self.status('preparing image; gradient is a placeholder')
        if self.preview:
            scale = 256 / max(width, height)
            image = ImageOps.colorize(Image.linear_gradient('L').resize((max(1, round(width*scale)), max(1, round(height*scale)))), '#202530', '#c5cbd4')
            self.publish(image)

    def publish(self, image):
        self.shared.state.assign_current_image(image)
        self.shared.state.current_image_sampling_step = self.shared.state.sampling_step

    def advance(self, step):
        step = min(self.steps + 1, max(self.completed, step))
        for _ in range(step - self.completed):
            self.shared.total_tqdm.update()
        self.completed = step
        self.shared.state.sampling_step = step
        self.shared.state.preview_step = step

    def update(self, event):
        if event['type'] == 'status':
            self.status(event['text'])
        elif event['type'] == 'progress':
            self.advance(int(event['step']))
            self.status(f"step {self.completed}/{self.steps}; preview is approximate")
            if event.get('preview') and self.preview:
                with Image.open(io.BytesIO(base64.b64decode(event['preview']))) as image:
                    self.publish(image.copy())

    def finish(self, image):
        self.advance(self.steps + 1)
        self.publish(image)
        self.status('image complete')

    def close(self):
        self.shared.total_tqdm.clear()
