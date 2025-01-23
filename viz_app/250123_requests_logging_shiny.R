#source("/Users/nikolas/Desktop/vlachos_project/250123_requests_logging_shiny.R")
library(shiny)
library(ggplot2)
library(dplyr)
library(jsonlite)
library(lubridate)
library(DT)
library(tidyr)

# Load and preprocess data
data <- fromJSON("data.json") %>% 
  as_tibble() %>%
  mutate(timestamp = ymd_hms(timestamp)) %>%
  # Process rankings and quality scores separately with unique prefixes
  unnest_wider(rankings, names_sep = "_rank_") %>%
  unnest_wider(quality_scores, names_sep = "_qual_") %>%
  # Pivot both metrics to long format
  pivot_longer(
    cols = starts_with("rankings_rank_"),
    names_to = "model",
    names_prefix = "rankings_rank_",
    values_to = "rankings"
  ) %>%
  left_join(
    pivot_longer(
      .,
      cols = starts_with("quality_scores_qual_"),
      names_to = "model_qual",
      names_prefix = "quality_scores_qual_",
      values_to = "quality_scores"
    ) %>%
      select(request_id, model_qual, quality_scores),
    by = "request_id"
  ) %>%
  filter(model == model_qual) %>%
  select(-model_qual)

# UI
ui <- navbarPage("LLM Quality Dashboard",
  theme = bslib::bs_theme(bootswatch = "minty"),
  
  tabPanel("Model Performance",
    sidebarLayout(
      sidebarPanel(
        selectInput("metric", "Select Metric:", 
                   choices = c("Rankings", "Quality Scores")),
        selectInput("model", "Focus Model:", 
                   choices = unique(data$model)),
        checkboxGroupInput("compare", "Compare Models:",
                          choices = unique(data$model),
                          selected = "claude"),
        actionButton("update", "Apply Filters", class = "btn-primary"),
        downloadButton("download_data", "Download Data")
      ),
      
      mainPanel(
        plotOutput("dist_plot", height = "400px"),
        plotOutput("time_plot", height = "400px"),
        DTOutput("stats_table")
      )
    )
  ),
  
  tabPanel("Study Summaries",
    DTOutput("summary_table"),
    plotOutput("summary_length_plot", height = "300px")
  )
)

# Server
server <- function(input, output) {
  
  filtered_data <- eventReactive(input$update, {
    data %>%
      filter(model %in% input$compare) %>%
      mutate(date = as_date(timestamp))
  })
  
  output$dist_plot <- renderPlot({
    df <- filtered_data()
    
    if(input$metric == "Rankings") {
        ggplot(df, aes(x = factor(rankings), fill = model)) +
            geom_bar(position = "dodge") +
            scale_fill_brewer(palette = "Set2") +
            labs(title = "Model Rankings Distribution", 
                 x = "Rank Position", y = "Count") +
            theme_minimal(base_size = 14)
    } else {
        ggplot(df, aes(x = factor(quality_scores), fill = model)) +
            geom_bar(position = "dodge") +
            scale_fill_brewer(palette = "Set2") +
            labs(title = "Quality Scores Distribution", 
                 x = "Quality Score", y = "Count") +
            theme_minimal(base_size = 14) +
            scale_x_discrete(drop = FALSE)  # Show all possible scores even if missing
    }
  })
  
  output$time_plot <- renderPlot({
    df <- filtered_data()
    
    df %>%
      group_by(date, model) %>%
      summarise(
        avg_rank = mean(rankings),
        avg_score = mean(quality_scores),
        .groups = "drop"
      ) %>%
      ggplot(aes(x = date, y = avg_rank, color = model)) +
      geom_line(linewidth = 1) +
      geom_point(size = 2) +
      scale_color_brewer(palette = "Set2") +
      labs(title = "Daily Average Rankings Trend", 
           x = "Date", y = "Average Rank") +
      theme_minimal(base_size = 14)
  })
  
  output$stats_table <- renderDT({
    filtered_data() %>%
      group_by(model) %>%
      summarise(
        `Average Rank` = round(mean(rankings), 2),
        `Average Score` = round(mean(quality_scores), 2),
        `Total Entries` = n(),
        .groups = "drop"
      ) %>%
      datatable(options = list(dom = 't'), rownames = FALSE)
  })
  
  output$summary_table <- renderDT({
    data %>%
      distinct(request_id, .keep_all = TRUE) %>%
      select(timestamp, summaries) %>%
      mutate(summary_count = map_int(summaries, length)) %>%
      datatable(options = list(pageLength = 5, scrollX = TRUE))
  })
  
  output$summary_length_plot <- renderPlot({
    data %>%
      distinct(request_id, .keep_all = TRUE) %>%
      mutate(summary_count = map_int(summaries, length)) %>%
      ggplot(aes(x = summary_count)) +
      geom_bar(fill = "#66c2a5") +
      labs(title = "Number of Summaries per Request", 
           x = "Number of Summaries", y = "Count") +
      theme_minimal(base_size = 14)
  })
  
  output$download_data <- downloadHandler(
    filename = function() {
      paste0("llm_analysis_data_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(filtered_data(), file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)