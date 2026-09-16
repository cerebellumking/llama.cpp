// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.paddle.ocr.util

import android.content.Context
import android.util.Log
import java.io.File
import org.opencv.android.OpenCVLoader

object OpenCVUtils {

    private var initialized = false
    @Volatile
    var lastError: String? = null
        private set

    @Synchronized
    fun init(context: Context): Boolean {
        if (initialized) return true
        lastError = null
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val errors = mutableListOf<String>()

        fun tryLoadLibrary(name: String): Boolean {
            return try {
                System.loadLibrary(name)
                true
            } catch (e: UnsatisfiedLinkError) {
                val path = File(nativeDir, "lib$name.so")
                errors += "$name loadLibrary: ${e.message}"
                if (!path.isFile) {
                    errors += "$name missing at $path"
                    return false
                }
                try {
                    System.load(path.absolutePath)
                    true
                } catch (e2: UnsatisfiedLinkError) {
                    errors += "$name load($path): ${e2.message}"
                    false
                }
            }
        }

        tryLoadLibrary("c++_shared")
        val ok = try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            errors += "OpenCVLoader.initLocal: ${t.message}"
            false
        } || tryLoadLibrary("opencv_java4")

        if (ok) {
            initialized = true
            return true
        }
        val listing = File(nativeDir).list()?.sorted()?.joinToString(",") ?: "(unreadable)"
        lastError = "nativeDir=$nativeDir files=[$listing] ${errors.joinToString(" | ")}"
        Log.e("OpenCVUtils", lastError!!)
        return false
    }
}
