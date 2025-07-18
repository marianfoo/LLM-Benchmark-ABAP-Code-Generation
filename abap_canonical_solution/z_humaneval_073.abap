CLASS z_humaneval_073 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: smallest_change
      IMPORTING
        arr           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_073 IMPLEMENTATION.

  METHOD smallest_change.
    DATA: ans TYPE i,
          len TYPE i,
          i   TYPE i.

    len = lines( arr ).
    ans = 0.

    DO len / 2 TIMES.
      i = sy-index - 1.
      IF arr[ i + 1 ] <> arr[ len - i ].
        ans = ans + 1.
      ENDIF.
    ENDDO.

    result = ans.
  ENDMETHOD.

ENDCLASS.