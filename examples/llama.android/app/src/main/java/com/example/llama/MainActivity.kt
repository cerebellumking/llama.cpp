package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.app.DownloadManager
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
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import androidx.core.net.toUri
import com.example.llama.api.ApiType
import com.example.llama.ui.theme.LlamaAndroidTheme
import java.io.File
import java.io.FileOutputStream
import com.benjaminwan.ocrlibrary.OcrEngine
import kotlinx.coroutines.delay
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.platform.LocalDensity

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

            // 调用OCR引擎，maxSideLen自适应图片分辨率，最大不超过2048
            val maxSideLen = minOf(maxOf(bitmap.width, bitmap.height), 2048)
            val result = ocrEngine.detect(bitmap, outputBitmap, maxSideLen)
            val allText = result.strRes

            // 清洗OCR结果
            var cleanedText = allText
                // 通用医师/医生/技师字段
                .replace(Regex("""\s*(?:[\u4e00-\u9fa5]{1,6})?(医师|医生|技师)[:：]?\s*[\u4e00-\u9fa5]{2,4}""")) { matchResult ->
                    val prefix = matchResult.value.replace(Regex("""[:：]?\s*[\u4e00-\u9fa5]{2,4}$"""), "")
                    ""
                }
                // 其他常见字段
                .replace(Regex("""姓名[:：]?\s*[\u4e00-\u9fa5]{2,4}"""), "")
                .replace(Regex("""患者[:：]?\s*[\u4e00-\u9fa5]{2,4}"""), "")
                // 替换身份证号
                .replace(Regex("\\d{17}[\\dXx]"), "")
                // 替换手机号
                .replace(Regex("1[3-9]\\d{9}"), "")
                // 替换住址
                .replace(Regex("地址[:：]?[\\u4e00-\\u9fa5A-Za-z0-9\\-]{4,}"), "")
                // 替换医院名
                .replace(Regex("[\\u4e00-\\u9fa5]{2,20}医院"), "")
                .replace(Regex("\\s+"), " ") // 将多个空白字符替换为单个空格
                .replace(Regex("[^\\p{L}\\p{N}\\p{P}\\s]"), "") // 只保留字母、数字、标点和空白字符
                .trim()

            // 你可以根据实际OCR内容继续扩展正则

            // 输出OCR结果用于调试
            viewModel.log("OCR原始结果：$allText")
            viewModel.log("OCR清洗后结果：$cleanedText")

            // 先添加图片消息
            viewModel.addImageMessage(bitmap)

            // 直接发送OCR文本给AI，不显示在界面上
            viewModel.updateMessage("请解读病例报告并给出简短建议：" + cleanedText)
            viewModel.send()

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
            // API 模型
            Downloadable(
                name = "DeepSeek API",
                source = null,  // API 模型不需要下载
                destination = null,  // API 模型不需要本地文件
                isApiModel = true
            ),
            Downloadable(
                name = "Qwen2.5-32B API",
                source = null,  // API 模型不需要下载
                destination = null,  // API 模型不需要本地文件
                isApiModel = true
            ),
            // 本地模型
            Downloadable(
                name = "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (Q4_K_M, 0.4 GiB)",
                source = Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf?download=true"),
                destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf"),
                isApiModel = false
            ),
            Downloadable(
                name = "DeepSeek-R1-DRAFT-Qwen2.5-0.5B (FP16, 1 GiB)",
                source = Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf?download=true"),
                destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf"),
                isApiModel = false
            ),
            // 协同推理
            Downloadable(
                name = "Qwen2.5-32B+Qwen2.5-0.5B_Q4_K_M",
                source = Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf?download=true"),
                destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-Q4_K_M.gguf"),
                isHetero = true
            ),
            Downloadable(
                name = "Qwen2.5-32B+Qwen2.5-0.5B",
                source = Uri.parse("https://huggingface.co/alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-GGUF/resolve/main/DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf?download=true"),
                destination = File(extFilesDir, "DeepSeek-R1-DRAFT-Qwen2.5-0.5B-f16.gguf"),
                isHetero = true
            )
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

        // 推理速度显示
        if (viewModel.inferenceSpeed > 0) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF2C2C2C))
                    .padding(8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "推理速度: %.1f tokens/s".format(viewModel.inferenceSpeed),
                    color = Color.White,
                    fontSize = 14.sp
                )
            }
        }

        // 聊天区域
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(Color.White)
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
                        isUserInput = chatMessage.type == MessageType.USER,
                        image = chatMessage.image
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
    var currentModelName by remember { mutableStateOf("Qwen") }

    // 用于锚定DropdownMenu
    var buttonCoords by remember { mutableStateOf<androidx.compose.ui.geometry.Rect?>(null) }

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
                            viewModel.load(model.destination!!.path, model.isHetero)
                            currentModelName = model.name
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

    Surface(color = Color(0xFF6200EE)) {
        Box(Modifier.fillMaxWidth().height(56.dp)) {
            val density = LocalDensity.current
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.align(Alignment.Center)
            ) {
                Text(
                    text = "HeteroSpec",
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(start = 24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                // 下拉按钮
                Box {
                    IconButton(
                        onClick = { showMenu = true },
                        modifier = Modifier
                            .size(28.dp)
                            .padding(top = 8.dp)
                            .offset(x = (-8).dp)
                            .onGloballyPositioned { coords ->
                                buttonCoords = coords.boundsInWindow()
                            }
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowDropDown,
                            contentDescription = "选择模型",
                            tint = Color.White
                        )
                    }
                    DropdownMenu(
                        expanded = showMenu,
                        onDismissRequest = { showMenu = false },
                        offset = buttonCoords?.let {
                            DpOffset((-220).dp, with(density) { it.height.toDp() - 35.dp })
                        } ?: DpOffset((-220).dp, (-35).dp),
                        modifier = Modifier
                            .background(Color(0xFFF9F9F9))
                            .width(320.dp)
                    ) {
                        // API模型分组
                        Text(
                            text = "API模型",
                            color = Color.Black,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                        )
                        Divider(color = Color.Gray.copy(alpha = 0.5f))
                        models.filter { it.isApiModel }.forEach { model ->
                            DropdownMenuItem(
                                text = { Text(model.name, color = Color.Black) },
                                onClick = {
                                    viewModel.clear()
                                    viewModel.switchToApiMode(
                                        when (model.name) {
                                            "DeepSeek API" -> ApiType.DEEPSEEK
                                            "Qwen2.5-32B API" -> ApiType.QWEN
                                            else -> ApiType.DEEPSEEK
                                        }
                                    )
                                    currentModelName = model.name
                                    showMenu = false
                                },
                                trailingIcon = {
                                    if (currentModelName == model.name) {
                                        Icon(
                                            Icons.Default.Check,
                                            contentDescription = "已选择",
                                            tint = Color.Black
                                        )
                                    }
                                }
                            )
                        }
                        // 本地模型分组
                        Divider(color = Color.Gray.copy(alpha = 0.5f))
                        Text(
                            text = "本地模型",
                            color = Color.Black,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                        )
                        Divider(color = Color.Gray.copy(alpha = 0.5f))
                        models.filter { !it.isApiModel and !it.isHetero }.forEach { model ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(model.name, color = Color.Black)
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
                                                color = Color.Black,
                                                fontSize = 12.sp,
                                                modifier = Modifier.padding(top = 4.dp)
                                            )
                                        }
                                    }
                                },
                                onClick = {
                                    if (downloadingModel == null) {
                                        if (model.destination?.exists() == true) {
                                            viewModel.clear() // 清除之前的对话
                                            viewModel.load(model.destination.path)
                                            currentModelName = model.name
                                            showMenu = false
                                        } else {
                                            val request = DownloadManager.Request(model.source!!)
                                                .setTitle(model.name)
                                                .setDescription("正在下载模型...")
                                                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                                                .setDestinationUri(model.destination!!.toUri())
                                            downloadId = dm.enqueue(request)
                                            downloadingModel = model
                                            viewModel.log("开始下载模型：${model.name}")
                                        }
                                    }
                                },
                                enabled = downloadingModel == null,
                                trailingIcon = {
                                    if (currentModelName == model.name) {
                                        Icon(
                                            Icons.Default.Check,
                                            contentDescription = "已选择",
                                            tint = Color.Black
                                        )
                                    }
                                }
                            )
                            if (models.indexOf(model) < models.filter { !it.isApiModel }.size - 1) {
                                Divider(color = Color.Gray.copy(alpha = 0.5f))
                            }
                        }
                        // 推测解码
                        Divider(color = Color.Gray.copy(alpha = 0.5f))
                        Text(
                            text = "推测解码",
                            color = Color.Black,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                        )
                        Divider(color = Color.Gray.copy(alpha = 0.5f))
                        models.filter { it.isHetero }.forEach { model ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(model.name, color = Color.Black)
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
                                                color = Color.Black,
                                                fontSize = 12.sp,
                                                modifier = Modifier.padding(top = 4.dp)
                                            )
                                        }
                                    }
                                },
                                onClick = {
                                    if (downloadingModel == null) {
                                        if (model.destination?.exists() == true) {
                                            viewModel.clear() // 清除之前的对话
                                            viewModel.load(model.destination.path, true)
                                            currentModelName = model.name
                                            showMenu = false
                                        } else {
                                            val request = DownloadManager.Request(model.source!!)
                                                .setTitle(model.name)
                                                .setDescription("正在下载模型...")
                                                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                                                .setDestinationUri(model.destination!!.toUri())
                                            downloadId = dm.enqueue(request)
                                            downloadingModel = model
                                            viewModel.log("开始下载草稿模型：${model.name}")
                                        }
                                    }
                                },
                                enabled = downloadingModel == null,
                                trailingIcon = {
                                    if (currentModelName == model.name) {
                                        Icon(
                                            Icons.Default.Check,
                                            contentDescription = "已选择",
                                            tint = Color.Black
                                        )
                                    }
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ImageViewerDialog(
    bitmap: Bitmap,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.8f)
                .background(Color.Black)
                .padding(16.dp)
        ) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Full size image",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )

            // 关闭按钮
            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close",
                    tint = Color.White
                )
            }
        }
    }
}

@Composable
fun MessageItem(content: String, isUserInput: Boolean, image: Bitmap? = null) {
    var showImageViewer by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalAlignment = if (isUserInput) Alignment.End else Alignment.Start
    ) {
        if (isUserInput) {
            // 用户输入使用对话框样式
            Box(
                modifier = Modifier
                    .widthIn(max = 280.dp)
                    .clip(
                        RoundedCornerShape(
                            topStart = 16.dp,
                            topEnd = 16.dp,
                            bottomStart = 16.dp,
                            bottomEnd = 4.dp
                        )
                    )
                    .background(Color(0xFF6200EE))
                    .padding(12.dp)
            ) {
                if (image != null) {
                    // 如果有图片，显示图片并添加点击事件
                    Image(
                        bitmap = image.asImageBitmap(),
                        contentDescription = "User image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 200.dp)
                            .clickable { showImageViewer = true },
                        contentScale = ContentScale.Fit
                    )
                } else {
                    // 如果没有图片，显示文本
                    Text(
                        text = content,
                        color = Color.White,
                        fontSize = 16.sp
                    )
                }
            }
        } else {
            // 系统输出使用普通文本样式
            Text(
                text = content,
                color = Color.Black,
                fontSize = 16.sp,
                modifier = Modifier.padding(horizontal = 16.dp)
            )
        }
    }

    // 显示图片查看器对话框
    if (showImageViewer && image != null) {
        ImageViewerDialog(
            bitmap = image,
            onDismiss = { showImageViewer = false }
        )
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
