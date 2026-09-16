plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.llama"
    compileSdk = 36
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.example.llama.aichat"

        minSdk = 33
        targetSdk = 36

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        debug {
            // JNI looks up IntVar.inc() by name. Debug minify previously stripped
            // that method (Kotlin never calls it) and aborted the app on send.
            isMinifyEnabled = false
            isShrinkResources = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    kotlin {
        jvmToolchain(17)
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

val alignedLibcxxDir = layout.buildDirectory.dir("generated/alignedJniLibs")

fun ndkLlvmPrebuilt(ndkHome: java.io.File): java.io.File {
    val root = ndkHome.resolve("toolchains/llvm/prebuilt")
    val hosts = listOf("darwin-x86_64", "darwin-aarch64", "linux-x86_64", "linux-aarch64")
    return hosts.map { root.resolve(it) }.firstOrNull { it.isDirectory }
        ?: error("NDK llvm prebuilt not found under $root")
}

val copyAlignedLibcxx = tasks.register<Copy>("copyAlignedLibcxx") {
    val libRoot = ndkLlvmPrebuilt(android.ndkDirectory).resolve("sysroot/usr/lib")
    into(alignedLibcxxDir)
    from(libRoot.resolve("aarch64-linux-android/libc++_shared.so")) {
        into("arm64-v8a")
    }
    from(libRoot.resolve("x86_64-linux-android/libc++_shared.so")) {
        into("x86_64")
    }
}

android.sourceSets.getByName("main").jniLibs.srcDir(alignedLibcxxDir)

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(copyAlignedLibcxx)
}

dependencies {
    implementation(libs.bundles.androidx)
    implementation(libs.material)
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.09.03"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.camera:camera-core:1.5.1")
    implementation("androidx.camera:camera-camera2:1.5.1")
    implementation("androidx.camera:camera-lifecycle:1.5.1")
    implementation("androidx.camera:camera-view:1.5.1")
    implementation("com.github.CanHub:Android-Image-Cropper:4.3.2")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation(project(":lib"))
    implementation(project(":PaddleOCR4Android"))
    implementation(project(":ppocr-sdk"))

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform("androidx.compose:compose-bom:2024.09.03"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
