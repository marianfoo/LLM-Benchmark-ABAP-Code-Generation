CLASS z_humaneval_029 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS filter_by_prefix
      IMPORTING
        VALUE(strings) TYPE string_table
        VALUE(prefix)  TYPE string
      RETURNING
        VALUE(result)  TYPE string_table.

ENDCLASS.

CLASS z_humaneval_029 IMPLEMENTATION.

  METHOD filter_by_prefix.
    DATA: lv_string     TYPE string,
          lv_prefix_len TYPE i.

    lv_prefix_len = strlen( prefix ).

    LOOP AT strings INTO lv_string.
      " Check if string starts with the given prefix
      IF strlen( lv_string ) >= lv_prefix_len AND
         lv_string+0(lv_prefix_len) = prefix.
        APPEND lv_string TO result.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.
