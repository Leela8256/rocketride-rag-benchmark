package com.rocketride.tika_api;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.concurrent.Executors;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpExchange;
import org.apache.tika.config.TikaConfig;
import org.apache.tika.io.TikaInputStream;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.parser.pdf.PDFParserConfig;
import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.sax.BodyContentHandler;
import com.rocketride.tika_api.EmbeddedContentExtractor.EmbeddedContentProcessor;

/** Shared parser backend for the comparison. No native engine or model inference.
 * Uses the tested engine's exact Tika jars, config, PDF flags, embedded extractor,
 * and StructureContentHandler. The writer mirrors NativeWriter's callback
 * boundaries and preprocessor_langchain.IInstance.writeText's newline separator.
 */
public class BenchTextServer {
    static TikaConfig config;
    static final long FLAGS = (1L << 24) | (1L << 25) | (1L << 26);

    static final class FramedWriter extends Writer {
        final char[] buffer = new char[1024 * 1024];
        final StringBuilder text = new StringBuilder();
        int length;
        int frames;
        void emit(char[] chars, int off, int len) {
            if (len == 0) return;
            text.append(Normalizer.normalize(new String(chars, off, len), Normalizer.Form.NFC));
            text.append('\n');
            frames++;
        }
        @Override public void write(char[] chars, int off, int len) {
            if (len > buffer.length - length) {
                flush();
                if (len > buffer.length) {
                    emit(chars, off, len);
                    return;
                }
            }
            System.arraycopy(chars, off, buffer, length, len);
            length += len;
        }
        @Override public void flush() {
            emit(buffer, 0, length);
            length = 0;
        }
        @Override public void close() { flush(); }
    }

    static void reply(HttpExchange exchange, int status, String text) throws IOException {
        byte[] body = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=UTF-8");
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) { out.write(body); }
    }

    static void parse(HttpExchange exchange) throws IOException {
        if (!exchange.getRequestMethod().equals("PUT")) { reply(exchange, 405, "PUT required"); return; }
        try (TikaInputStream input = TikaInputStream.get(exchange.getRequestBody())) {
            Metadata metadata = new Metadata();
            metadata.set(Metadata.CONTENT_TYPE, "application/pdf");
            String contentLength = exchange.getRequestHeaders().getFirst("Content-Length");
            if (contentLength != null) metadata.set(Metadata.CONTENT_LENGTH, contentLength);
            if (contentLength == null || Long.parseLong(contentLength) > 8L * 1024 * 1024) input.getPath();
            AutoDetectParser parser = new AutoDetectParser(config);
            String mime = parser.getDetector().detect(input, metadata).toString();
            ParseContext context = new ParseContext();
            context.set(Parser.class, parser);
            context.set(PDFParserConfig.class, TikaApi.getPdfConfig(FLAGS));
            context.set(EmbeddedDocumentExtractor.class, new EmbeddedContentProcessor(0, mime, FLAGS));
            FramedWriter writer = new FramedWriter();
            StructureContentHandler handler = new StructureContentHandler(new BodyContentHandler(writer), metadata);
            parser.parse(input, handler, metadata, context);
            writer.close();
            exchange.getResponseHeaders().set("X-Text-Frames", Integer.toString(writer.frames));
            reply(exchange, 200, writer.text.toString());
        } catch (Exception error) {
            error.printStackTrace();
            reply(exchange, 500, error.toString());
        }
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("java.awt.headless", "true");
        TikaApi.init();
        config = TikaApi.getTikaConfig();
        int port = Integer.getInteger("bench.port", 9998);
        HttpServer server = HttpServer.create(new InetSocketAddress(System.getProperty("bench.host", "127.0.0.1"), port), 128);
        server.createContext("/version", ex -> reply(ex, 200, "RocketRide parser jars; NativeWriter-compatible text framing"));
        server.createContext("/tika", BenchTextServer::parse);
        server.setExecutor(Executors.newFixedThreadPool(32));
        server.start();
        System.out.println("READY document-preparation parser on " + port);
    }
}
