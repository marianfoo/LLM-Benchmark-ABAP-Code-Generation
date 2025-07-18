CLASS z_humaneval_014 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.

    CLASS-METHODS: all_prefixes
      IMPORTING string        TYPE string
      RETURNING VALUE(result) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_014 IMPLEMENTATION.

  METHOD all_prefixes.
    DATA: lv_string TYPE string,
          lt_result TYPE string_table,
          lv_prefix TYPE string,
          lv_index  TYPE i.

    lv_string = string.

    DO strlen( lv_string ) TIMES.
      lv_index = sy-index.
      lv_prefix = lv_string(lv_index).
      APPEND lv_prefix TO lt_result.
    ENDDO.

    result = lt_result.
  ENDMETHOD.

ENDCLASS.