package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.provider.MediaStore
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.lifecycle.lifecycleScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import com.canhub.cropper.CropImageContract
import com.canhub.cropper.CropImageContractOptions
import com.canhub.cropper.CropImageOptions
import com.example.llama.ocr.DualOcrController
import com.example.llama.ocr.OcrBackend
import com.example.llama.ui.theme.LlamaAndroidTheme
import com.example.llama.ui.workspace.ReportWorkspaceRoute
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.launch

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
) : ComponentActivity() {
    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    @Suppress("unused")
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()
    private lateinit var dualOcr: DualOcrController

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            takePicture()
        }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val imageBitmap = result.data?.extras?.get("data") as? Bitmap
            imageBitmap?.let { bitmap ->
                val uri = saveBitmapToCache(bitmap)
                startCrop(uri)
            }
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                startCrop(uri)
            } catch (e: Exception) {
                viewModel.log("图片处理失败：${e.message}")
            }
        }
    }

    private val cropImageLauncher = registerForActivityResult(CropImageContract()) { result ->
        if (result.isSuccessful) {
            val croppedImageUri = result.uriContent
            croppedImageUri?.let {
                val inputStream = contentResolver.openInputStream(it)
                val croppedBitmap = BitmapFactory.decodeStream(inputStream)
                croppedBitmap?.let { processImage(it) }
            }
        } else {
            viewModel.log("裁剪失败：${result.error?.message}")
        }
    }

    private fun processImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            try {
                viewModel.log("开始识别文字...")
                val rawText = dualOcr.recognize(bitmap)
                val cleanedText = cleanOcrText(rawText)
                viewModel.log("OCR原始结果：$rawText")
                viewModel.log("OCR清洗后结果：$cleanedText")
                viewModel.addImageMessage(bitmap)
                viewModel.showOcrEditor(cleanedText, bitmap)
            } catch (e: Exception) {
                viewModel.log("文字识别失败：${e.message}")
            }
        }
    }

    private fun takePicture() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        takePictureLauncher.launch(intent)
    }

    fun pickImage() {
        pickImageLauncher.launch("image/*")
    }

    fun ocrEngine(): DualOcrController = dualOcr

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> takePicture()
            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    fun startCamera() {
        checkCameraPermission()
    }

    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    private fun startCrop(uri: Uri) {
        val cropOptions = CropImageOptions().apply {
            allowCounterRotation = true
            allowFlipping = true
            allowRotation = true
            cropShape = com.canhub.cropper.CropImageView.CropShape.RECTANGLE
            guidelines = com.canhub.cropper.CropImageView.Guidelines.ON
            fixAspectRatio = false
            showCropOverlay = true
            outputCompressFormat = Bitmap.CompressFormat.JPEG
            outputCompressQuality = 90
        }
        cropImageLauncher.launch(CropImageContractOptions(uri, cropOptions))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        LLamaAndroid.instance().setNativeLibraryDir(applicationInfo.nativeLibraryDir)

        val preferences = getSharedPreferences(OcrBackend.PREFS, MODE_PRIVATE)
        dualOcr = DualOcrController(this, preferences, viewModel::log)
        viewModel.log("正在加载OCR模型…")
        lifecycleScope.launch {
            try {
                dualOcr.start()
            } catch (e: Exception) {
                viewModel.log("OCR模型加载失败: $e")
            }
        }

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        val models = defaultModels(getExternalFilesDir(null))

        setContent {
            LlamaAndroidTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ReportWorkspaceRoute(
                        viewModel = viewModel,
                        dm = downloadManager,
                        models = models,
                        ocr = dualOcr,
                        onCamera = { startCamera() },
                        onPickImage = { pickImage() },
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        dualOcr.release()
    }

    private fun saveBitmapToCache(bitmap: Bitmap): Uri {
        val file = File(cacheDir, "temp_${System.currentTimeMillis()}.jpg")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
        }
        return Uri.fromFile(file)
    }
}

fun defaultModels(extFilesDir: File?): List<Downloadable> = listOf(
    Downloadable(
        name = "DeepSeek API",
        source = null,
        destination = null,
        isApiModel = true,
    ),
    Downloadable(
        name = "Qwen2.5-32B API",
        source = null,
        destination = null,
        isApiModel = true,
    ),
    Downloadable(
        name = "Qwen3-32B API",
        source = null,
        destination = null,
        isApiModel = true,
    ),
    Downloadable(
        name = "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (Q4_K_M, 0.4 GiB)",
        source = Uri.parse("https://modelscope.cn/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf?download=true"),
        destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf"),
    ),
    Downloadable(
        name = "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (FP16, 1 GiB)",
        source = Uri.parse("https://modelscope.cn/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf?download=true"),
        destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf"),
    ),
    Downloadable(
        name = "Qwen3-0.6B (Q4_0, 0.4 GiB)",
        source = Uri.parse("https://modelscope.cn/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true"),
        destination = File(extFilesDir, "Qwen3-0.6B-Q4_0.gguf"),
    ),
    Downloadable(
        name = "Qwen3-0.6B (FP16, 1.2GiB)",
        source = Uri.parse("https://modelscope.cn/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf?download=true"),
        destination = File(extFilesDir, "Qwen3-0.6B-BF16.gguf"),
    ),
    Downloadable(
        name = "Qwen2.5-32B+Qwen2.5-0.5B_Q4_K_M",
        source = Uri.parse("https://modelscope.cn/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf?download=true"),
        destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf"),
        isHetero = true,
    ),
    Downloadable(
        name = "Qwen2.5-32B+Qwen2.5-0.5B",
        source = Uri.parse("https://modelscope.cn/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf?download=true"),
        destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf"),
        isHetero = true,
    ),
    Downloadable(
        name = "Qwen3-32B+Qwen3-0.6B_Q4_0",
        source = Uri.parse("https://modelscope.cn/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true"),
        destination = File(extFilesDir, "Qwen3-0.6B-Q4_0.gguf"),
        isHetero = true,
    ),
    Downloadable(
        name = "Qwen3-32B+Qwen3-0.6B",
        source = Uri.parse("https://modelscope.cn/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf?download=true"),
        destination = File(extFilesDir, "Qwen3-0.6B-BF16.gguf"),
        isHetero = true,
    ),
)
