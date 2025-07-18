CLASS ltcl_test_cycpattern_check DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_0 FOR TESTING,
      test_1 FOR TESTING,
      test_2 FOR TESTING,
      test_3 FOR TESTING,
      test_4 FOR TESTING,
      test_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_cycpattern_check IMPLEMENTATION.

  METHOD test_0.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'xyzw' b = 'xyw' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false msg = 'test #0' ).
  ENDMETHOD.

  METHOD test_1.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'yello' b = 'ell' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true msg = 'test #1' ).
  ENDMETHOD.

  METHOD test_2.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'whattup' b = 'ptut' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false msg = 'test #2' ).
  ENDMETHOD.

  METHOD test_3.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'efef' b = 'fee' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true msg = 'test #3' ).
  ENDMETHOD.

  METHOD test_4.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'abab' b = 'aabb' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false msg = 'test #4' ).
  ENDMETHOD.

  METHOD test_5.
    DATA: result TYPE abap_bool.
    result = z_humaneval_154=>cycpattern_check( a = 'winemtt' b = 'tinem' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true msg = 'test #5' ).
  ENDMETHOD.

ENDCLASS.