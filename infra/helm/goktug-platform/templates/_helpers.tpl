{{/*
Common helpers for goktug-platform.
*/}}

{{- define "goktug.labels" -}}
app.kubernetes.io/part-of: goktug-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
Per-service selector labels. Call with a dict: (dict "name" $name "root" $).
*/}}
{{- define "goktug.selectorLabels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
{{- end -}}

{{/*
Resolve the container image for a service.
Call with: (dict "name" $name "svc" $svc "root" $)
Precedence: per-service image override > global registry/tag.
*/}}
{{- define "goktug.image" -}}
{{- $g := .root.Values.global -}}
{{- if .svc.image -}}
{{ .svc.image }}
{{- else -}}
{{ $g.imageRegistry }}/{{ .name }}:{{ default $g.imageTag .svc.imageTag }}
{{- end -}}
{{- end -}}

{{/*
Liveness/readiness path resolution: java services use actuator probe paths,
others use an explicit healthPath. Call with (dict "svc" $svc "root" $ "kind" "liveness|readiness").
*/}}
{{- define "goktug.healthPath" -}}
{{- $d := .root.Values.defaults -}}
{{- if .svc.java -}}
{{- if eq .kind "liveness" }}{{ $d.livenessPath }}{{ else }}{{ $d.readinessPath }}{{ end -}}
{{- else -}}
{{ default "/health" .svc.healthPath }}
{{- end -}}
{{- end -}}
