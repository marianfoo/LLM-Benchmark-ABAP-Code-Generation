CLASS z_humaneval_081 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_grade,
             gpa TYPE f,
           END OF ty_grade.
    TYPES: tt_grades TYPE TABLE OF ty_grade.

    CLASS-METHODS: numerical_letter_grade
      IMPORTING
        grades               TYPE tt_grades
      RETURNING
        VALUE(letter_grades) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_081 IMPLEMENTATION.

  METHOD numerical_letter_grade.
    DATA: lv_grade TYPE string.

    LOOP AT grades INTO DATA(ls_grade).
      IF ls_grade-gpa = '4.0'.
        lv_grade = 'A+'.
      ELSEIF ls_grade-gpa > '3.7'.
        lv_grade = 'A'.
      ELSEIF ls_grade-gpa > '3.3'.
        lv_grade = 'A-'.
      ELSEIF ls_grade-gpa > '3.0'.
        lv_grade = 'B+'.
      ELSEIF ls_grade-gpa > '2.7'.
        lv_grade = 'B'.
      ELSEIF ls_grade-gpa > '2.3'.
        lv_grade = 'B-'.
      ELSEIF ls_grade-gpa > '2.0'.
        lv_grade = 'C+'.
      ELSEIF ls_grade-gpa > '1.7'.
        lv_grade = 'C'.
      ELSEIF ls_grade-gpa > '1.3'.
        lv_grade = 'C-'.
      ELSEIF ls_grade-gpa > '1.0'.
        lv_grade = 'D+'.
      ELSEIF ls_grade-gpa > '0.7'.
        lv_grade = 'D'.
      ELSEIF ls_grade-gpa > '0.0'.
        lv_grade = 'D-'.
      ELSE.
        lv_grade = 'E'.
      ENDIF.

      letter_grades = VALUE #( BASE letter_grades ( lv_grade ) ).
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.