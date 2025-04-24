package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.provider.MediaStore
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import java.io.File
import java.io.FileOutputStream
import com.benjaminwan.ocrlibrary.OcrEngine

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()
    private lateinit var ocrEngine: OcrEngine

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
                processImage(bitmap)
            }
        }
    }

    private fun processImage(bitmap: Bitmap) {
        try {
            // 创建临时文件保存图片
            val tempFile = File(cacheDir, "temp_image.jpg")
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }

            // 创建输出图片
            val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)

            // 调用OCR引擎
            val result = ocrEngine.detect(bitmap, outputBitmap, 1024)
            val allText = result.strRes
            viewModel.log("识别到的文字：$allText")
        } catch (e: Exception) {
            viewModel.log("文字识别失败：${e.message}")
        }
    }

    private fun takePicture() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        takePictureLauncher.launch(intent)
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                takePicture()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    fun startCamera() {
        checkCameraPermission()
    }

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化OCR引擎
        ocrEngine = OcrEngine(applicationContext)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)

        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        val extFilesDir = getExternalFilesDir(null)

        val models = listOf(
            Downloadable(
                "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (Q4_K_M, 0.4 GiB)",
                Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf?download=true"),
                File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf"),
            ),
            Downloadable(
                "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (FP16, 1 GiB)",
                Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf?download=true"),
                File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf"),
            ),
            // Downloadable(
            //     "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
            //     Uri.parse("https://huggingface.co/TheBloke/phi-2-dpo-GGUF/resolve/main/phi-2-dpo.Q3_K_M.gguf?download=true"),
            //     File(extFilesDir, "phi-2-dpo.Q3_K_M.gguf")
            // ),
        )

        setContent {
            LlamaAndroidTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainCompose(
                        viewModel,
                        clipboardManager,
                        downloadManager,
                        models,
                        this
                    )
                }
            }
        }
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    dm: DownloadManager,
    models: List<Downloadable>,
    activity: MainActivity
) {
    Column {
        val scrollState = rememberLazyListState()

        Box(modifier = Modifier.weight(1f)) {
            LazyColumn(state = scrollState) {
                items(viewModel.messages) {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodyLarge.copy(color = LocalContentColor.current),
                        modifier = Modifier.padding(16.dp)
                    )
                }
            }
        }
        OutlinedTextField(
            value = viewModel.message,
            onValueChange = { viewModel.updateMessage(it) },
            label = { Text("Message") },
        )
        Row {
            Button({ viewModel.send() }) { Text("Send") }
            Button({ viewModel.bench(8, 4, 1) }) { Text("Bench") }
            Button({ viewModel.clear() }) { Text("Clear") }
            Button({
                viewModel.messages.joinToString("\n").let {
                    clipboard.setPrimaryClip(ClipData.newPlainText("", it))
                }
            }) { Text("Copy") }
            Button({ activity.startCamera() }) { Text("拍照") }
        }

        Column {
            for (model in models) {
                Downloadable.Button(viewModel, dm, model)
            }
        }
    }
}
