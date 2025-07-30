# 🔐 Google Cloud Storage Authentication Setup

## Option 1: Service Account Key (Recommended for Scripts)

### Step 1: Create Service Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Click **Create Service Account**
4. Name it: `vr-sports-uploader`
5. Add description: `Upload videos and models to GCS`
6. Click **Create and Continue**

### Step 2: Grant Permissions
1. Select these roles:
   - **Storage Object Admin** (for uploading files)
   - **Storage Object Viewer** (for reading files)
2. Click **Continue** → **Done**

### Step 3: Create Key File
1. Click on your service account
2. Go to **Keys** tab
3. Click **Add Key** → **Create New Key**
4. Choose **JSON** format
5. Download the key file
6. Save it as `gcs-key.json` in your project root

### Step 4: Set Environment Variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcs-key.json"
```

## Option 2: Application Default Credentials

### Step 1: Install Google Cloud SDK
```bash
# Download and install from:
# https://cloud.google.com/sdk/docs/install
```

### Step 2: Authenticate
```bash
gcloud auth application-default login
```

## Option 3: Manual Upload (Quick Test)

### Step 1: Upload via Web Console
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Cloud Storage** → **Buckets**
3. Click on your `still_data` bucket
4. Click **Upload Files**
5. Drag and drop your videos

### Step 2: Test the Upload Script
```bash
# Install the dependency
pip install google-cloud-storage

# Test with a small file first
python3 upload_to_gcs.py --file test_images/test_0.jpg --destination test/test_0.jpg
```

## 🔧 Troubleshooting

### "Default credentials not found"
- Make sure you've set up authentication (Option 1, 2, or 3)
- Check that `GOOGLE_APPLICATION_CREDENTIALS` is set correctly

### "Permission denied"
- Ensure your service account has the right permissions
- Check that the bucket name is correct

### "Bucket not found"
- Verify the bucket name: `still_data`
- Make sure you're in the right Google Cloud project

## 🚀 Quick Start (Recommended)

1. **Use Option 1** (Service Account Key) for the most reliable setup
2. **Download the JSON key file** and save as `gcs-key.json`
3. **Set the environment variable**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcs-key.json"
   ```
4. **Test the upload**:
   ```bash
   python3 upload_to_gcs.py --list
   ```

## 📁 File Structure After Setup
```
VR_DEV/
├── gcs-key.json              # Service account key (keep secure!)
├── upload_to_gcs.py          # Upload script
├── requirements.txt           # Dependencies
└── ... (other project files)
```

## ⚠️ Security Note
- Keep your `gcs-key.json` file secure
- Add it to `.gitignore` to prevent accidental commits
- Never share the key file publicly 