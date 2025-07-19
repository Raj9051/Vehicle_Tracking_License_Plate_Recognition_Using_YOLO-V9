def process_image(path, args, i):
    config = dict(
        threshold_d=args.detection_threshold,
        threshold_o=args.ocr_threshold,
        mode="redaction",
    )

    # Predictions
    source_im = Image.open(path)
    if source_im.mode != "RGB":
        source_im = source_im.convert("RGB")
    images = [((0, 0), source_im)]  # Entire image
    # Top left and top right crops
    if args.split_image:
        y = 0
        win_size = 0.55
        width, height = source_im.width * win_size, source_im.height * win_size
        for x in [0, int((1 - win_size) * source_im.width)]:
            images.append(((x, y), source_im.crop((x, y, x + width, y + height))))

    # Inference
    results = []
    for (x, y), im in images:
        im_bytes = io.BytesIO()
        im.save(im_bytes, "JPEG", quality=95)
        im_bytes.seek(0)
        im_results = recognition_api(
            im_bytes, args.regions, args.api_key, args.sdk_url, config=config
        )
        results.append(dict(prediction=im_results, x=x, y=y))
    results = post_processing(merge_results(results))
    results["filename"] = Path(path).name

    # Set bounding box padding
    for item in results["results"]:
        # Decrease padding size for large bounding boxes
        b = item["box"]
        width, height = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
        padding_x = int(max(0, width * (0.3 * math.exp(-10 * width / source_im.width))))
        padding_y = int(
            max(0, height * (0.3 * math.exp(-10 * height / source_im.height)))
        )
        b["xmin"] = b["xmin"] - padding_x
        b["ymin"] = b["ymin"] - padding_y
        b["xmax"] = b["xmax"] + padding_x
        b["ymax"] = b["ymax"] + padding_y

    if args.show_boxes or args.save_blurred:
        im = blur(
            source_im,
            5,
            results,
            ignore_no_bb=args.ignore_no_bb,
            ignore_list=args.ignore_regexp,
        )

        if args.show_boxes:
            im.show()
        if args.save_blurred:
            filename = Path(path)
            im.save(filename.parent / (f"{filename.stem}_blurred{filename.suffix}"))
    if 0:
        draw_bb(source_im, results["results"]).show()
    return results

def blur(im, blur_amount, api_res, ignore_no_bb=False, ignore_list=None):
    for res in api_res.get("results", []):
        if ignore_no_bb and res["vehicle"]["score"] == 0.0:
            continue

        if ignore_list:
            skip_blur = False
            for ignore_regex in ignore_list:
                if re.search(ignore_regex, res["plate"]):
                    skip_blur = True
                    break
            if skip_blur:
                continue

        b = res["box"]
        width, height = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
        crop_box = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
        ic = im.crop(crop_box)

        # Increase amount of blur with size of bounding box
        blur_image = ic.filter(
            ImageFilter.GaussianBlur(
                radius=math.sqrt(width * height) * 0.3 * blur_amount / 10
            )
        )
        im.paste(blur_image, crop_box)
    return im
