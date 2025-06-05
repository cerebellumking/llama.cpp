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
import androidx.compose.ui.graphics.Brush
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
import dev.jeziellago.compose.markdowntext.MarkdownText
import androidx.compose.runtime.snapshotFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive

// 优化的配色方案
object AppColors {
    // 主色调 - 优雅的蓝色系
    val Primary = Color(0xFF2B7DE9)
    val PrimaryDark = Color(0xFF1B5FC7)
    val PrimaryLight = Color(0xFF4A90FF)
    
    // 背景色 - 现代化渐变
    val Background = Color(0xFFF7F9FC)
    val BackgroundSecondary = Color(0xFFFFFFFF)
    val BackgroundDark = Color(0xFF0F1419)
    
    // 聊天气泡
    val UserBubble = Color(0xFF2B7DE9)
    val AIBubble = Color(0xFFFFFFFF)
    val AIBubbleBorder = Color(0xFFE8ECF1)
    
    // 文本颜色
    val TextPrimary = Color(0xFF1A1A1A)
    val TextSecondary = Color(0xFF6B7280)
    val TextWhite = Color(0xFFFFFFFF)
    val TextMuted = Color(0xFF9CA3AF)
    
    // 输入框
    val InputBackground = Color(0xFFFFFFFF)
    val InputBorder = Color(0xFFE5E7EB)
    val InputBorderFocused = Color(0xFF2B7DE9)
    
    // 应用栏 - 优雅的渐变蓝色
    val AppBarBackground = Color(0xFF2B7DE9)
    val AppBarBackgroundSecondary = Color(0xFF4A90FF)
    val StatusBarBackground = Color(0xFFFFFFFF)
    val StatusBarBackgroundDark = Color(0xFF1A1A1A)
    
    // 分割线
    val Divider = Color(0xFFE5E7EB)
    
    // 阴影
    val Shadow = Color(0x0A000000)
}

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

            // 调用OCR引擎，maxSideLen自适应图片分辨率
            val maxSideLen = minOf(maxOf(bitmap.width, bitmap.height), 4096)
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
                    color = AppColors.Background
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
            .background(AppColors.Background)
    ) {
        // 应用栏
        AppBar(models, viewModel, dm)

        // 推理速度显示
        if (viewModel.inferenceSpeed > 0) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                AppColors.Primary.copy(alpha = 0.12f),
                                AppColors.PrimaryLight.copy(alpha = 0.08f)
                            )
                        )
                    )
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Speed,
                        contentDescription = "推理速度",
                        tint = AppColors.Primary,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "推理速度: %.1f tokens/s".format(viewModel.inferenceSpeed),
                        color = AppColors.Primary,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }

        // 聊天区域
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(AppColors.Background)
        ) {
            val scrollState = rememberLazyListState()

            LaunchedEffect(Unit) {
                while (isActive) {
                    delay(1000L)
                    snapshotFlow {
                        Pair(viewModel.messages.size, viewModel.messages.lastOrNull()?.content)
                    }
                        .collectLatest { (currentSize, _) ->
                            scrollState.animateScrollToItem(currentSize)
                        }
                }
            }

            LazyColumn(
                state = scrollState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
                contentPadding = PaddingValues(vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
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

    // 应用栏 - 使用渐变背景
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        AppColors.AppBarBackground,
                        AppColors.AppBarBackgroundSecondary
                    )
                )
            )
    ) {
        val density = LocalDensity.current
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .align(Alignment.Center)
                .padding(horizontal = 16.dp)
        ) {
            Text(
                text = "HeteroSpec",
                color = AppColors.TextWhite,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.width(8.dp))
            // 下拉按钮
            Box {
                IconButton(
                    onClick = { showMenu = true },
                    modifier = Modifier
                        .size(32.dp)
                        .onGloballyPositioned { coords ->
                            buttonCoords = coords.boundsInWindow()
                        }
                ) {
                    Icon(
                        imageVector = Icons.Default.ArrowDropDown,
                        contentDescription = "选择模型",
                        tint = AppColors.TextWhite
                    )
                }
                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false },
                    offset = buttonCoords?.let {
                        DpOffset((-220).dp, with(density) { it.height.toDp() - 35.dp })
                    } ?: DpOffset((-220).dp, (-35).dp),
                    modifier = Modifier
                        .background(AppColors.BackgroundSecondary)
                        .width(320.dp)
                ) {
                    // API模型分组
                    Text(
                        text = "API模型",
                        color = AppColors.TextPrimary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                    Divider(color = AppColors.Divider)
                    models.filter { it.isApiModel }.forEach { model ->
                        DropdownMenuItem(
                            text = { 
                                Text(
                                    model.name, 
                                    color = AppColors.TextPrimary,
                                    fontSize = 14.sp
                                ) 
                            },
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
                                        tint = AppColors.Primary
                                    )
                                }
                            }
                        )
                    }
                    // 本地模型分组
                    Divider(color = AppColors.Divider)
                    Text(
                        text = "本地模型",
                        color = AppColors.TextPrimary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                    Divider(color = AppColors.Divider)
                    models.filter { !it.isApiModel and !it.isHetero }.forEach { model ->
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(
                                        model.name, 
                                        color = AppColors.TextPrimary,
                                        fontSize = 14.sp
                                    )
                                    if (downloadingModel == model) {
                                        LinearProgressIndicator(
                                            progress = downloadProgress.toFloat(),
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(top = 8.dp),
                                            color = AppColors.Primary,
                                            trackColor = AppColors.Divider
                                        )
                                        Text(
                                            text = "${(downloadProgress * 100).toInt()}%",
                                            color = AppColors.TextSecondary,
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
                                        tint = AppColors.Primary
                                    )
                                }
                            }
                        )
                    }
                    // 推测解码
                    Divider(color = AppColors.Divider)
                    Text(
                        text = "推测解码",
                        color = AppColors.TextPrimary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                    Divider(color = AppColors.Divider)
                    models.filter { it.isHetero }.forEach { model ->
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(
                                        model.name, 
                                        color = AppColors.TextPrimary,
                                        fontSize = 14.sp
                                    )
                                    if (downloadingModel == model) {
                                        LinearProgressIndicator(
                                            progress = downloadProgress.toFloat(),
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(top = 8.dp),
                                            color = AppColors.Primary,
                                            trackColor = AppColors.Divider
                                        )
                                        Text(
                                            text = "${(downloadProgress * 100).toInt()}%",
                                            color = AppColors.TextSecondary,
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
                                        tint = AppColors.Primary
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
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUserInput) Alignment.End else Alignment.Start
    ) {
        if (isUserInput) {
            if (image != null) {
                // 图片消息 - 直接使用背景色
                Box(
                    modifier = Modifier
                        .widthIn(max = 280.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(AppColors.Background)
                        .clickable { showImageViewer = true }
                        .padding(8.dp)
                ) {
                    Image(
                        bitmap = image.asImageBitmap(),
                        contentDescription = "User image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 200.dp)
                            .clip(RoundedCornerShape(12.dp)),
                        contentScale = ContentScale.Fit
                    )
                }
            } else {
                // 文本消息 - 使用蓝色背景
                Box(
                    modifier = Modifier
                        .widthIn(max = 280.dp)
                        .clip(
                            RoundedCornerShape(
                                topStart = 20.dp,
                                topEnd = 20.dp,
                                bottomStart = 20.dp,
                                bottomEnd = 4.dp
                            )
                        )
                        .background(AppColors.UserBubble)
                        .padding(16.dp)
                ) {
                    Text(
                        text = content,
                        color = AppColors.TextWhite,
                        fontSize = 16.sp,
                        lineHeight = 22.sp
                    )
                }
            }
        } else {
            // AI回复气泡 - 简洁边框设计
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(AppColors.AIBubble)
                    .padding(12.dp)
            ) {
                val markdownContent = remember(content) { content }
                MarkdownText(
                    markdown = markdownContent,
                    modifier = Modifier.fillMaxWidth(),
                    style = MaterialTheme.typography.bodyLarge.copy(
                        color = AppColors.TextPrimary,
                        fontSize = 16.sp,
                        lineHeight = 24.sp
                    )
                )
            }
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
    Column {
        // 分割线
        Divider(
            color = AppColors.Divider,
            thickness = 1.dp
        )
        
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(AppColors.BackgroundSecondary)
                .padding(16.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            // 相机按钮
            IconButton(
                onClick = onCamera,
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(AppColors.Background)
            ) {
                Icon(
                    imageVector = Icons.Default.Camera,
                    contentDescription = "Take Photo",
                    tint = AppColors.Primary,
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.width(8.dp))

            // 相册按钮
            IconButton(
                onClick = { activity.pickImage() },
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(AppColors.Background)
            ) {
                Icon(
                    imageVector = Icons.Default.Image,
                    contentDescription = "Pick Image",
                    tint = AppColors.Primary,
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            // 输入框
            OutlinedTextField(
                value = viewModel.message,
                onValueChange = { viewModel.updateMessage(it) },
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(24.dp)),
                placeholder = { 
                    Text(
                        "输入消息...",
                        color = AppColors.TextMuted,
                        fontSize = 16.sp
                    ) 
                },
                shape = RoundedCornerShape(24.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = AppColors.InputBorderFocused,
                    unfocusedBorderColor = AppColors.InputBorder,
                    focusedContainerColor = AppColors.InputBackground,
                    unfocusedContainerColor = AppColors.InputBackground,
                    focusedTextColor = AppColors.TextPrimary,
                    unfocusedTextColor = AppColors.TextPrimary
                ),
                maxLines = 4
            )

            Spacer(modifier = Modifier.width(12.dp))

            // 发送按钮
            Button(
                onClick = onSend,
                modifier = Modifier
                    .height(56.dp)
                    .clip(RoundedCornerShape(28.dp)),
                colors = ButtonDefaults.buttonColors(
                    containerColor = AppColors.Primary,
                    disabledContainerColor = AppColors.TextMuted
                ),
                contentPadding = PaddingValues(horizontal = 24.dp)
            ) {
                Text(
                    "发送",
                    color = AppColors.TextWhite,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}
