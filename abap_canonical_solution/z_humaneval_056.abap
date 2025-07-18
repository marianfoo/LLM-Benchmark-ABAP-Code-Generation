CLASS z_humaneval_056 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: correct_bracketing
      IMPORTING
        VALUE(brackets) TYPE string
      RETURNING
        VALUE(result)   TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_056 IMPLEMENTATION.

  METHOD correct_bracketing.
    DATA: depth    TYPE i VALUE 0,
          lv_char  TYPE c LENGTH 1,
          lv_index TYPE i.

    DO strlen( brackets ) TIMES.
      lv_index = sy-index - 1.
      lv_char = brackets+lv_index(1).
      IF lv_char = '<'.
        depth = depth + 1.
      ELSE.
        depth = depth - 1.
      ENDIF.
      IF depth < 0.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    IF depth = 0.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.