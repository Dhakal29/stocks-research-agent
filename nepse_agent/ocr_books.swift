// Local OCR for scanned PDF pages using frameworks included with macOS.
import Foundation
import PDFKit
import Vision

guard CommandLine.arguments.count == 4,
      let document = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else {
    fputs("Usage: ocr_books.swift input.pdf output.json page_numbers\n", stderr)
    exit(1)
}
let pages = CommandLine.arguments[3].split(separator: ",").compactMap { Int($0) }
var results: [String: String] = [:]
do {
    for (offset, number) in pages.enumerated() {
        try autoreleasepool {
            guard let page = document.page(at: number - 1) else {
                throw NSError(domain: "BookOCR", code: 1)
            }
            let bounds = page.bounds(for: .mediaBox)
            let scale: CGFloat = 2.5
            guard let context = CGContext(
                data: nil, width: Int(bounds.width * scale), height: Int(bounds.height * scale),
                bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { throw NSError(domain: "BookOCR", code: 2) }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: CGFloat(context.width), height: CGFloat(context.height)))
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            page.draw(with: .mediaBox, to: context)
            guard let image = context.makeImage() else { throw NSError(domain: "BookOCR", code: 3) }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["en-US"]
            request.usesLanguageCorrection = true
            try VNImageRequestHandler(cgImage: image).perform([request])
            results[String(number)] = (request.results ?? []).compactMap {
                $0.topCandidates(1).first?.string
            }.joined(separator: "\n")
        }
        if (offset + 1) % 10 == 0 || offset + 1 == pages.count {
            fputs("OCR: \(offset + 1)/\(pages.count) pages processed\n", stderr)
        }
    }
    let data = try JSONSerialization.data(withJSONObject: results, options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
} catch {
    fputs("Book OCR failed: \(error.localizedDescription)\n", stderr)
    exit(1)
}
