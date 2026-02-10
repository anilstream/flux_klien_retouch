# Standard library
import logging
import tempfile
import time
from io import BytesIO

# Standard typing
from typing import Annotated, Literal

# Third-party
from PIL import Image
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import Field, HttpUrl
import uvicorn

# Local application
from klien_model import FluxKlienMaskedInpaint, FluxKlienGenFill
from klien_utils import fetch_image_data, get_outpaint_padding, resize_image, get_white_mask


masked_inpainter = FluxKlienMaskedInpaint()
image_outpainter = FluxKlienGenFill()

templates = Jinja2Templates(directory="templates")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(process)d - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
app = FastAPI(
    title="Retouch API",
    version="1.0.0",
    openapi_url="/retouch/openapi.json",
    docs_url="/retouch/docs",
    redoc_url="/retouch/redoc",
)

@app.get('/retouch/status')
def retouch_preset_status_get(request: Request):
    return {'status': 'OK'}


@app.get("/retouch/klien/inpaint/generate")
def retouch_preset_predict_get(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/retouch/klien/inpaint/generate")
async def retouch_preset_predict_post(request: Request, image: UploadFile = File(None), mask: UploadFile = File(None),
                                      image_url: HttpUrl = Form(None), mask_url: HttpUrl = Form(None),
                                      prompt: str = Form(...), invert: bool = Form(False)):
    try:
        t1 = time.perf_counter()
        logger.info(f"prompt: {prompt}")

        image = fetch_image_data(image_url) if image_url  else await image.read()
        mask =  fetch_image_data(mask_url) if mask_url else await mask.read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image, \
                tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_mask:

            temp_image.write(image)
            temp_image.flush()

            temp_mask.write(mask)
            temp_mask.flush()

            output = masked_inpainter.run(temp_image.name, temp_mask.name, prompt, mask_invert=invert)

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}

@app.post("/retouch/klien/edit/generate")
async def retouch_preset_predict_post(request: Request, image: UploadFile = File(None),
                                      image_url: HttpUrl = Form(None), prompt: str = Form(...)):
    try:
        t1 = time.perf_counter()
        logger.info(f"prompt: {prompt}")

        image = fetch_image_data(image_url) if image_url  else await image.read()
        mask = get_white_mask(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image, \
                tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_mask:

            temp_image.write(image)
            temp_image.flush()

            temp_mask.write(mask)
            temp_mask.flush()

            output = masked_inpainter.run(temp_image.name, temp_mask.name, prompt)

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}

@app.post("/retouch/klien/outpaint/generate")
async def genfill_preset_predict_post(request: Request, image: UploadFile = File(None), image_url: HttpUrl = Form(None),
                                      height: int = Form(None), width: int = Form(None),
                                      top: int = Form(0), bottom: int = Form(0), left: int = Form(0), right: int = Form(0)):
    try:
        t1 = time.perf_counter()
        logger.info(f"top: {top}, bottom: {bottom}, left: {left}, right: {right}, width: {width}, height: {height}")

        image = fetch_image_data(image_url) if image_url  else await image.read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image:

            temp_image.write(image)
            temp_image.flush()

            if height and width:
                padded = get_outpaint_padding(temp_image.name, (width,height))
                logger.info(f"padding: {padded}")
                print(f"padding: {padded}")

                resized = padded['resized']
                resized.save(temp_image.name)
                left, right, top, bottom = padded["left"], padded["right"], padded["top"], padded["bottom"]
                output = image_outpainter.run(temp_image.name, top=top, bottom=bottom, left=left, right=right)
            else:
                output = image_outpainter.run(temp_image.name, top=top,bottom=bottom, left=left, right=right)
            print(f"Output size: {Image.open(BytesIO(output)).size}")

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        # resize to exact resolution
        if height and width:
             output = resize_image(output, (width, height))

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("klien_api:app", host="0.0.0.0", port=5007, workers=1)
