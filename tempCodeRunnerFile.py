@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'error': 'Không có ảnh'}), 400

        img_bytes = base64.b64decode(img_base64.split(',')[1])
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Không decode được ảnh'}), 400

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        name, best_sim = "Unknown", -1.0
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin*w), int(bbox.ymin*h)
                bw, bh = int(bbox.width*w), int(bbox.height*h)
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x+bw), min(h, y+bh)
                face_crop = frame[y:y2, x:x2]
                if face_crop is None or face_crop.size == 0:
                    continue
                emb = get_embedding_from_bgr(face_crop)
                for k_emb, k_name in zip(known_embeddings, known_names):
                    sim = float(cosine_similarity([emb], [k_emb])[0][0])
                    if sim > best_sim:
                        best_sim = sim
                        name = k_name if sim >= SIM_THRESHOLD else "Unknown"

        print(f"📸 Kết quả: {name} ({best_sim:.2f})")
        return jsonify({"recognized": name, "similarity": best_sim, "name": name})
    except Exception as e:
        print("⚠️ Lỗi xử lý:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = (data.get('name') or "Unknown").strip()
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'status': 'error', 'message': 'Không có ảnh'}), 400

        timestamp = int(time.time())
        safe_name = name.replace(' ', '_')
        filename = os.path.join(UPLOADS_DIR, f"{safe_name}_{timestamp}.jpg")
        img_bytes = base64.b64decode(img_base64.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(img_bytes)

        drive_id = None
        if DRIVE_AVAILABLE:
            drive_id = upload_file_to_drive(filename, title=os.path.basename(filename))
            if drive_id:
                print(f"✅ Uploaded {filename} to Drive id={drive_id}")
            else:
                print("ℹ️ Upload to Drive failed or returned None.")

        ok = load_sample(filename, name)
        if not ok:
            return jsonify({'status': 'error', 'message': 'Không thể tạo embedding từ ảnh đã tải lên.'}), 500

        save_registered_entry(name, filename, drive_id)

        return jsonify({'status': 'success', 'message': f'Đăng ký thành công: {name}', 'file': filename, 'drive_id': drive_id})
    except Exception as e:
        print("⚠️ Lỗi register:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500