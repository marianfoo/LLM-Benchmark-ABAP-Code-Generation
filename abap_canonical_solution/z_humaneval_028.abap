CLASS z_humaneval_028 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS concatenate
      IMPORTING
        VALUE(strings) TYPE string_table
      RETURNING
        VALUE(result)  TYPE string.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_028 IMPLEMENTATION.
  METHOD concatenate.
    CLEAR result.
    LOOP AT strings INTO DATA(string).
      CONCATENATE result string INTO result.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.