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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import androidx.core.net.toUri
import com.example.llama.ui.theme.LlamaAndroidTheme
import java.io.File
import java.io.FileOutputStream
import com.benjaminwan.ocrlibrary.OcrEngine
import kotlinx.coroutines.delay

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

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                processImage(bitmap)
            } catch (e: Exception) {
                viewModel.log("图片处理失败：${e.message}")
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

    fun pickImage() {
        pickImageLauncher.launch("image/*")
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
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1A1A1A))
    ) {
        // 应用栏
        AppBar(models, viewModel, dm)

        // 聊天区域
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            val scrollState = rememberLazyListState()
            
            LazyColumn(
                state = scrollState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp)
            ) {
                items(viewModel.messages) { chatMessage ->
                    MessageItem(
                        content = chatMessage.content,
                        isUserInput = chatMessage.type == MessageType.USER
                    )
                }
            }
        }

        // 输入区域
        InputArea(
            viewModel = viewModel,
            onSend = { viewModel.send() },
            onCamera = { activity.startCamera() },
            activity = activity
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppBar(
    models: List<Downloadable>,
    viewModel: MainViewModel,
    dm: DownloadManager
) {
    var showMenu by remember { mutableStateOf(false) }
    var downloadingModel by remember { mutableStateOf<Downloadable?>(null) }
    var downloadProgress by remember { mutableDoubleStateOf(0.0) }
    var downloadId by remember { mutableLongStateOf(-1L) }

    // 监听下载进度
    LaunchedEffect(downloadId) {
        if (downloadId != -1L) {
            while (true) {
                val cursor = dm.query(DownloadManager.Query().setFilterById(downloadId))
                if (cursor != null && cursor.moveToFirst()) {
                    val bytesDownloaded = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                    val bytesTotal = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                    downloadProgress = bytesDownloaded.toDouble() / bytesTotal
                    
                    if (bytesDownloaded == bytesTotal) {
                        // 下载完成后加载模型
                        downloadingModel?.let { model ->
                            viewModel.clear() // 清除之前的对话
                            viewModel.load(model.destination.path)
                        }
                        downloadingModel = null
                        downloadId = -1L
                        downloadProgress = 0.0
                        break
                    }
                }
                cursor?.close()
                delay(1000)
            }
        }
    }

    TopAppBar(
        title = {
            Text(
                text = "RouteLLM",
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
        },
        navigationIcon = {
            IconButton(onClick = { showMenu = !showMenu }) {
                Icon(
                    imageVector = Icons.Default.Menu,
                    contentDescription = "Menu",
                    tint = Color.White
                )
            }
        },
        actions = {
            IconButton(onClick = { /* TODO */ }) {
                Icon(
                    imageVector = Icons.Default.Headset,
                    contentDescription = "Headphone",
                    tint = Color.White
                )
            }
            IconButton(onClick = { /* TODO */ }) {
                Icon(
                    imageVector = Icons.Default.MoreVert,
                    contentDescription = "More",
                    tint = Color.White
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Color(0xFF6200EE)
        )
    )

    if (showMenu) {
        DropdownMenu(
            expanded = showMenu,
            onDismissRequest = { showMenu = false },
            modifier = Modifier
                .background(Color(0xFF2C2C2C))
                .width(300.dp)
        ) {
            Text(
                text = "可用模型",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(16.dp)
            )
            Divider(color = Color.Gray)
            models.forEach { model ->
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(
                                text = model.name,
                                color = Color.White
                            )
                            if (downloadingModel == model) {
                                LinearProgressIndicator(
                                    progress = downloadProgress.toFloat(),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(top = 8.dp),
                                    color = Color(0xFF6200EE),
                                    trackColor = Color.Gray
                                )
                                Text(
                                    text = "${(downloadProgress * 100).toInt()}%",
                                    color = Color.White,
                                    fontSize = 12.sp,
                                    modifier = Modifier.padding(top = 4.dp)
                                )
                            }
                        }
                    },
                    onClick = {
                        if (downloadingModel == null) {
                            if (model.destination.exists()) {
                                // 如果模型已下载，直接加载
                                viewModel.clear() // 清除之前的对话
                                viewModel.load(model.destination.path)
                                showMenu = false
                            } else {
                                // 否则开始下载
                                val request = DownloadManager.Request(model.source)
                                    .setTitle(model.name)
                                    .setDescription("正在下载模型...")
                                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                                    .setDestinationUri(model.destination.toUri())
                                downloadId = dm.enqueue(request)
                                downloadingModel = model
                                viewModel.log("开始下载模型：${model.name}")
                            }
                        }
                    },
                    enabled = downloadingModel == null
                )
            }
        }
    }
}

@Composable
fun MessageItem(content: String, isUserInput: Boolean) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalAlignment = if (isUserInput) Alignment.End else Alignment.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUserInput) 16.dp else 4.dp,
                        bottomEnd = if (isUserInput) 4.dp else 16.dp
                    )
                )
                .background(
                    if (isUserInput) Color(0xFF6200EE)
                    else Color(0xFFF5F5F5)
                )
                .padding(12.dp)
        ) {
            Text(
                text = content,
                color = if (isUserInput) Color.White else Color.Black,
                fontSize = 16.sp
            )
        }
    }
}

@Composable
fun InputArea(
    viewModel: MainViewModel,
    onSend: () -> Unit,
    onCamera: () -> Unit,
    activity: MainActivity
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.White)
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // 相机按钮
        IconButton(onClick = onCamera) {
            Icon(
                imageVector = Icons.Default.Camera,
                contentDescription = "Take Photo",
                tint = Color(0xFF6200EE)
            )
        }
        
        // 相册按钮
        IconButton(onClick = { activity.pickImage() }) {
            Icon(
                imageVector = Icons.Default.Image,
                contentDescription = "Pick Image",
                tint = Color(0xFF6200EE)
            )
        }

        OutlinedTextField(
            value = viewModel.message,
            onValueChange = { viewModel.updateMessage(it) },
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 8.dp),
            placeholder = { Text("输入消息...") },
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF6200EE),
                unfocusedBorderColor = Color(0xFFE0E0E0)
            )
        )

        Button(
            onClick = onSend,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF6200EE)
            )
        ) {
            Text("发送")
        }
    }
}
