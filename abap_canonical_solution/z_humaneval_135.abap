CLASS z_humaneval_135 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS can_arrange
      IMPORTING
        VALUE(arr)    TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_135 IMPLEMENTATION.

  METHOD can_arrange.
    DATA: ind TYPE i VALUE -1,
          i   TYPE i VALUE 1.

    WHILE i < lines( arr ).
      IF arr[ i + 1 ] < arr[ i ].
        ind = i.
      ENDIF.
      i = i + 1.
    ENDWHILE.

    result = ind.
  ENDMETHOD.

ENDCLASS.