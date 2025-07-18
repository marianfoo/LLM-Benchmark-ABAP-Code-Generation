CLASS z_humaneval_061 DEFINITION
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

CLASS z_humaneval_061 IMPLEMENTATION.

  METHOD correct_bracketing.
    DATA: depth TYPE i VALUE 0,
          len   TYPE i,
          char  TYPE c LENGTH 1,
          index TYPE i.

    len = strlen( brackets ).

    DO len TIMES.
      index = sy-index - 1.
      char = brackets+index(1).
      IF char = '('.
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