package flask_tools.pipette.java;

import com.bioinceptionlabs.reactionblast.api.RDT;
import com.bioinceptionlabs.reactionblast.api.ReactionResult;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Batch-friendly CLI for mapping reaction SMILES from stdin or a file.
 *
 * <p>Each non-empty input line must be a single reaction SMILES containing
 * {@code >>}. The output is one JSON object per line, preserving input order.
 */
public final class PipetteAtomMapperCli {

    private PipetteAtomMapperCli() {}

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.showHelp) {
            printUsage();
            return;
        }

        List<String> reactions = readReactions(options);
        int emitted = 0;
        for (String rawReaction : reactions) {
            String reaction = rawReaction == null ? "" : rawReaction.trim();
            if (reaction.isEmpty()) {
                continue;
            }

            try {
                ReactionResult result = RDT.map(
                        reaction,
                        options.generate2D,
                        options.complexMapping
                );
                String mappedSmiles = result.getMappedSmiles();
                if (mappedSmiles == null || mappedSmiles.isBlank()) {
                    throw new IllegalStateException("RDT returned no mapped SMILES");
                }
                System.out.println(
                        buildRecord(
                                emitted,
                                reaction,
                                mappedSmiles,
                                result.getAlgorithm(),
                                result.getFormedCleavedCount(),
                                result.getOrderChangeCount(),
                                result.getStereoChangeCount(),
                                null
                        )
                );
            } catch (Exception exc) {
                System.out.println(
                        buildRecord(
                                emitted,
                                reaction,
                                null,
                                null,
                                null,
                                null,
                                null,
                                rootCauseMessage(exc)
                        )
                );
            }
            emitted++;
        }
    }

    private static List<String> readReactions(Options options) throws IOException {
        if (!options.positionalReactions.isEmpty()) {
            return options.positionalReactions;
        }

        if (options.inputPath != null) {
            return Files.readAllLines(options.inputPath, StandardCharsets.UTF_8);
        }

        List<String> reactions = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                reactions.add(line);
            }
        }
        return reactions;
    }

    private static String rootCauseMessage(Throwable throwable) {
        Throwable current = throwable;
        while (current.getCause() != null) {
            current = current.getCause();
        }
        String message = current.getMessage();
        if (message == null || message.isBlank()) {
            message = current.toString();
        }
        return current.getClass().getSimpleName() + ": " + message;
    }

    private static String buildRecord(
            int index,
            String inputSmiles,
            String mappedSmiles,
            String algorithm,
            Integer formedCleavedCount,
            Integer orderChangeCount,
            Integer stereoChangeCount,
            String error
    ) {
        return "{"
                + "\"index\":" + index + ","
                + "\"input_smiles\":" + jsonString(inputSmiles) + ","
                + "\"mapped_smiles\":" + jsonString(mappedSmiles) + ","
                + "\"algorithm\":" + jsonString(algorithm) + ","
                + "\"formed_cleaved_count\":" + jsonNumber(formedCleavedCount) + ","
                + "\"order_change_count\":" + jsonNumber(orderChangeCount) + ","
                + "\"stereo_change_count\":" + jsonNumber(stereoChangeCount) + ","
                + "\"error\":" + jsonString(error)
                + "}";
    }

    private static String jsonNumber(Integer value) {
        return value == null ? "null" : Integer.toString(value);
    }

    private static String jsonString(String value) {
        if (value == null) {
            return "null";
        }

        StringBuilder builder = new StringBuilder();
        builder.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    builder.append("\\\"");
                    break;
                case '\\':
                    builder.append("\\\\");
                    break;
                case '\b':
                    builder.append("\\b");
                    break;
                case '\f':
                    builder.append("\\f");
                    break;
                case '\n':
                    builder.append("\\n");
                    break;
                case '\r':
                    builder.append("\\r");
                    break;
                case '\t':
                    builder.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        builder.append(String.format("\\u%04x", (int) c));
                    } else {
                        builder.append(c);
                    }
            }
        }
        builder.append('"');
        return builder.toString();
    }

    private static void printUsage() {
        System.err.println("Usage: java -cp <helper-classes>:<rdt-jar> "
                + "flask_tools.pipette.java.PipetteAtomMapperCli "
                + "[--input reactions.txt] [--no-2d] [--simple-mapping] [reaction ...]");
        System.err.println(
                "Reads one reaction SMILES per line from stdin when no file or positional reactions are provided."
        );
    }

    private static final class Options {
        private final Path inputPath;
        private final boolean generate2D;
        private final boolean complexMapping;
        private final boolean showHelp;
        private final List<String> positionalReactions;

        private Options(
                Path inputPath,
                boolean generate2D,
                boolean complexMapping,
                boolean showHelp,
                List<String> positionalReactions
        ) {
            this.inputPath = inputPath;
            this.generate2D = generate2D;
            this.complexMapping = complexMapping;
            this.showHelp = showHelp;
            this.positionalReactions = positionalReactions;
        }

        private static Options parse(String[] args) {
            Path inputPath = null;
            boolean generate2D = true;
            boolean complexMapping = true;
            boolean showHelp = false;
            List<String> positional = new ArrayList<>();

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                switch (arg) {
                    case "-h":
                    case "--help":
                        showHelp = true;
                        break;
                    case "-i":
                    case "--input":
                        if (i + 1 >= args.length) {
                            throw new IllegalArgumentException("--input requires a file path");
                        }
                        inputPath = Path.of(args[++i]);
                        break;
                    case "--no-2d":
                        generate2D = false;
                        break;
                    case "--simple-mapping":
                        complexMapping = false;
                        break;
                    default:
                        positional.add(arg);
                        break;
                }
            }

            return new Options(inputPath, generate2D, complexMapping, showHelp, positional);
        }
    }
}
