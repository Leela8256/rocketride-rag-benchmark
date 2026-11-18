# The parser the LangGraph service calls: RocketRide's own Java parser, taken from the engine image, behind a
# small HTTP server. Both systems then turn a PDF into the same text, so the comparison starts after parsing with
# identical input. No part of the engine runs here, only its parser jars and their configuration.
ARG ENGINE_IMAGE=ragbench-rocketride:base

FROM ${ENGINE_IMAGE} AS engine

FROM eclipse-temurin:17-jdk AS build
COPY --from=engine /opt/rocketride/engine/java/lib /engine-lib
COPY BenchTextServer.java /src/BenchTextServer.java
RUN javac -proc:none --release 17 -cp "/engine-lib/*" -d /bench-classes /src/BenchTextServer.java

FROM ${ENGINE_IMAGE}
COPY --from=build /bench-classes /bench-classes
WORKDIR /opt/rocketride/engine
# Listens on 127.0.0.1:9998 unless -Dbench.host and -Dbench.port say otherwise.
ENTRYPOINT ["/opt/rocketride/engine/java/jre/bin/java", "-cp", "/opt/rocketride/engine/java/lib/*:/bench-classes", "com.rocketride.tika_api.BenchTextServer"]
CMD []
