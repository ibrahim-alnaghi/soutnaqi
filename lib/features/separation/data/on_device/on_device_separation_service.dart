import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:soutnaqi/core/config/app_env.dart';
import 'package:soutnaqi/core/errors/app_exception.dart';
import 'package:soutnaqi/core/logging/app_log.dart';
import 'package:soutnaqi/features/separation/data/on_device/audio_tensor_codec.dart';
import 'package:soutnaqi/features/separation/data/on_device/demucs_chunker.dart';
import 'package:soutnaqi/features/separation/data/on_device/on_device_model_repository.dart';
import 'package:soutnaqi/features/separation/data/on_device/on_device_model_spec.dart';
import 'package:soutnaqi/features/separation/data/on_device/onnx_inference_runner.dart';
import 'package:soutnaqi/features/separation/data/separation_audio_io.dart';
import 'package:soutnaqi/features/separation/data/separation_service.dart';
import 'package:soutnaqi/features/separation/data/separation_target.dart';
import 'package:uuid/uuid.dart';

SeparationService createOnDeviceSeparationService() =>
    OnDeviceSeparationService();

/// Fully offline separation via a Demucs model exported to ONNX
/// (see [OnDeviceModelSpec]). No server, no per-request network call — the
/// model is downloaded once and cached by [OnDeviceModelRepository].
class OnDeviceSeparationService implements SeparationService {
  OnDeviceSeparationService({OnDeviceModelRepository? modelRepository})
      : _modelRepository = modelRepository ?? OnDeviceModelRepository();

  static const _uuid = Uuid();

  final OnDeviceModelRepository _modelRepository;

  @override
  bool get isSupported => AppEnv.isOnDeviceSeparationSupported;

  @override
  Future<String> separate({
    required String inputAudioPath,
    required SeparationTarget target,
  }) async {
    if (!isSupported) {
      throw const AppException(messageKey: 'separationNotConfigured');
    }

    appLog.d('⚡ Starting on-device Demucs separation: $target');
    var preparedPath = inputAudioPath;
    OnnxInferenceRunner? runner;
    try {
      preparedPath = await SeparationAudioIo.prepareWavInput(inputAudioPath);
      await _modelRepository.ensureModelDownloaded();

      final mix = await AudioTensorCodec.decodeWav(preparedPath);
      runner = await OnnxInferenceRunner.load(
        await _modelRepository.modelPath(),
      );

      final stems = await DemucsChunker.process(
        mix: mix,
        runChunk: runner.runChunk,
      );

      final targetSamples = target == SeparationTarget.vocals
          ? stems[OnDeviceModelSpec.vocalsStemIndex]
          : _mixNonVocalStems(stems);

      final directory = await getTemporaryDirectory();
      final wavOutput = '${directory.path}/soutnaqi_${_uuid.v4()}.wav';
      await AudioTensorCodec.encodeWav(
        outputPath: wavOutput,
        samples: targetSamples,
      );

      final outputPath = await SeparationAudioIo.encodeWavToM4a(wavOutput);
      appLog.d('✅ On-device separation complete: $outputPath');
      return outputPath;
    } on AppException {
      rethrow;
    } catch (error) {
      appLog.e('❌ On-device separation failed', error: error);
      throw AppException(messageKey: 'separationFailed', cause: error);
    } finally {
      await runner?.dispose();
      if (preparedPath != inputAudioPath) {
        try {
          await File(preparedPath).delete();
        } catch (_) {}
      }
    }
  }

  /// The model predicts 4 stems ([OnDeviceModelSpec.sources]); the
  /// instrumental target is the sum of every stem except vocals.
  StereoSamples _mixNonVocalStems(List<StereoSamples> stems) {
    final length = stems[OnDeviceModelSpec.vocalsStemIndex].length;
    final left = Float32List(length);
    final right = Float32List(length);
    for (var s = 0; s < stems.length; s++) {
      if (s == OnDeviceModelSpec.vocalsStemIndex) continue;
      final stem = stems[s];
      for (var i = 0; i < length; i++) {
        left[i] += stem.left[i];
        right[i] += stem.right[i];
      }
    }
    return StereoSamples(left: left, right: right);
  }
}
